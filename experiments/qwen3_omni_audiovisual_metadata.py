#!/usr/bin/env python3
"""Probe Qwen3-Omni video-only and native audiovisual metadata behavior."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import sys
import time
from typing import Any

from ampav.qwen.omni import DEFAULT_MODEL_ID, Qwen3OmniTransformers


DEFAULT_PROMPT = """Analyze this media segment for archival discovery.
Return only one JSON object with keys summary, visible_evidence,
audible_evidence, audiovisual_events, subjects, themes, and limitations.
Each evidence item must be concise and use source \"visual\", \"audio\", or
\"both\". Report only evidence grounded in the supplied modalities. Do not infer
identities or protected attributes. Do not claim audio evidence when no audio was
supplied. Keep each array to at most 4 high-value items."""

MODES = {
    "video-only": False,
    "audiovisual": True,
}


def package_version(name: str) -> str | None:
    """Return an installed distribution version when available."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_safe(value: Any) -> Any:
    """Convert common native values to JSON-compatible data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (RuntimeError, ValueError):
            pass
    return str(value)


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse one JSON object, tolerating a surrounding Markdown fence."""

    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as error:
        return None, str(error)
    if not isinstance(value, dict):
        return None, "response JSON is not an object"
    return value, None


def parse_args() -> argparse.Namespace:
    """Parse bounded model, media, modality, and output settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODES),
        default=list(MODES),
    )
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--total-pixels", type=int, default=4_194_304)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--attention-implementation",
        default="sdpa",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    parser.add_argument("--max-memory-per-gpu", default="22GiB")
    return parser.parse_args()


def tensor_shape(value: Any) -> list[int] | None:
    """Return a tensor-like shape without retaining its values."""

    shape = getattr(value, "shape", None)
    return [int(item) for item in shape] if shape is not None else None


def main() -> None:
    """Run a controlled video-only versus audiovisual Qwen3-Omni pair."""

    args = parse_args()
    if not args.video_path.is_file():
        raise FileNotFoundError(args.video_path)
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.total_pixels < 1:
        raise ValueError("total-pixels must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    # The Eggplant runtime profile records Decord as the working H.264 backend.
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

    import torch
    import transformers
    import yaml
    from transformers.utils.hub import cached_file

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    gpu_count = torch.cuda.device_count()
    max_memory = {index: args.max_memory_per_gpu for index in range(gpu_count)}
    prompt = (
        args.prompt_file.read_text(encoding="utf-8")
        if args.prompt_file is not None
        else DEFAULT_PROMPT
    )
    (args.output_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "input_ref.txt").write_text(
        f"video: {args.video_path.resolve()}\n"
        "cleanup: caller-owned input was not modified or deleted\n",
        encoding="utf-8",
    )

    started_at = datetime.now().astimezone()
    wall_start = time.perf_counter()
    max_rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    load_start = time.perf_counter()
    tool = Qwen3OmniTransformers(
        args.model_id,
        revision=args.model_revision,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        model_load_options={"max_memory": max_memory},
    )
    model_load_seconds = time.perf_counter() - load_start

    results: dict[str, Any] = {}
    for mode in args.modes:
        use_audio = MODES[mode]
        for device_index in range(gpu_count):
            torch.cuda.reset_peak_memory_stats(device_index)

        prepare_start = time.perf_counter()
        inputs = tool.prepare(
            args.video_path,
            prompt,
            use_audio_in_video=use_audio,
            video_options={
                "fps": args.fps,
                "total_pixels": args.total_pixels,
            },
            processor_options={"cap_pixels_per_frame": True},
        )
        prepare_seconds = time.perf_counter() - prepare_start
        input_tokens = int(inputs["input_ids"].shape[-1])
        input_shapes = {
            str(key): shape
            for key, value in inputs.items()
            if (shape := tensor_shape(value)) is not None
        }

        generation_start = time.perf_counter()
        generated_ids = tool.generate(
            inputs,
            use_audio_in_video=use_audio,
            generation_options={"max_new_tokens": args.max_new_tokens},
        )
        for device_index in range(gpu_count):
            torch.cuda.synchronize(device_index)
        generation_seconds = time.perf_counter() - generation_start
        response_text = tool.decode(generated_ids, input_tokens)
        sequences = getattr(generated_ids, "sequences", generated_ids)
        output_tokens = int(sequences.shape[-1]) - input_tokens
        structured, parse_error = parse_json_response(response_text)

        results[mode] = {
            "use_audio_in_video": use_audio,
            "response_text": response_text,
            "structured": structured,
            "parse_error": parse_error,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_shapes": input_shapes,
            "prepare_seconds": prepare_seconds,
            "generation_seconds": generation_seconds,
            "cuda_peak_allocated_bytes": [
                torch.cuda.max_memory_allocated(index)
                for index in range(gpu_count)
            ],
            "cuda_peak_reserved_bytes": [
                torch.cuda.max_memory_reserved(index)
                for index in range(gpu_count)
            ],
        }
        (args.output_dir / "native_output.partial.json").write_text(
            json.dumps(
                json_safe(
                    {
                        "native_type": (
                            "transformers Qwen3OmniMoeThinker generation and decode"
                        ),
                        "completed_modes": list(results),
                        "results": results,
                    }
                ),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    completed_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - wall_start
    config_path = cached_file(
        args.model_id,
        "config.json",
        revision=args.model_revision,
    )
    resolved_revision = None
    if config_path:
        path = Path(config_path)
        if path.parent.parent.name == "snapshots":
            resolved_revision = path.parent.name

    native_output = {
        "native_type": "transformers Qwen3OmniMoeThinker generation and decode",
        "results": results,
    }
    manifest = {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "model_load_seconds": model_load_seconds,
        "model_id": args.model_id,
        "requested_model_revision": args.model_revision,
        "resolved_model_revision": resolved_revision,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": package_version("accelerate"),
            "qwen_omni_utils": package_version("qwen-omni-utils"),
            "decord": package_version("decord"),
            "av": package_version("av"),
            "cuda_runtime": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
            "model_device_map": json_safe(getattr(tool.model, "hf_device_map", None)),
        },
        "request": {
            "modes": args.modes,
            "fps": args.fps,
            "total_pixels": args.total_pixels,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "attention_implementation": args.attention_implementation,
            "max_memory": max_memory,
            "do_sample": False,
            "video_reader": os.environ["FORCE_QWENVL_VIDEO_READER"],
            "cap_pixels_per_frame": True,
        },
        "resource": {
            "max_rss_before_kib": max_rss_before_kib,
            "max_rss_after_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }

    (args.output_dir / "native_output.json").write_text(
        json.dumps(json_safe(native_output), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(json_safe(manifest), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (args.output_dir / "observations.md").write_text(
        "# Observations\n\n"
        "Review the paired native outputs for grounded visual-only evidence, "
        "new audible evidence, cross-modal events, contradictions, and unsupported "
        "claims. Runtime measurements are compatibility evidence from this run, "
        "not a general benchmark.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(native_output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
