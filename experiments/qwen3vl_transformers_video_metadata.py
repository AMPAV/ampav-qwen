#!/usr/bin/env python3
"""Probe Qwen3-VL video metadata with the native Transformers runtime.

This experiment intentionally keeps model loading, prompting, persistence, and
keyframe extraction outside the AMPAV tool API. It is an opt-in real-model
probe for comparing runtime behavior and metadata prompts.
"""

from __future__ import annotations

import argparse
from copy import copy
from datetime import datetime
from functools import partial
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import shlex
import sys
import time
from typing import Any

from ampav.qwen.vl import Qwen3VLTransformersVideo


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_MAX_TOTAL_VIDEO_PIXELS = 8_388_608
DEFAULT_MIN_TOTAL_VIDEO_PIXELS = 4096

COMMON_RULES = """Return only one JSON object matching the requested shape.
Use no markdown fences. Report only evidence grounded in the supplied inputs.
Do not infer identities or protected attributes. Use null rather than inventing
a timestamp. Keep each array to at most 8 high-value items. Every evidence item
must use source "visual", "transcript", or "both". Visual timestamps must be
grounded only in the video frames; transcript timestamps, if present, are
transcript evidence and must not be used to validate visual timing."""

TASK_SCHEMAS: dict[str, dict[str, Any]] = {
    "combined": {
        "aboutness": {
            "summary": "string",
            "subjects": ["string"],
            "themes": ["string"],
            "evidence": [
                {
                    "description": "string",
                    "source": "visual|transcript|both",
                    "timestamp_seconds": "number|null",
                }
            ],
        },
        "visible_text": [
            {
                "text": "string",
                "timestamp_seconds": "number|null",
                "source": "visual",
            }
        ],
        "people_and_objects": [
            {
                "description": "string",
                "kind": "person|object",
                "timestamp_seconds": "number|null",
                "source": "visual",
            }
        ],
        "activities_and_events": [
            {
                "description": "string",
                "start_seconds": "number|null",
                "end_seconds": "number|null",
                "source": "visual|transcript|both",
            }
        ],
        "shots": [
            {
                "start_seconds": "number|null",
                "end_seconds": "number|null",
                "description": "string",
                "transition": "string|null",
                "source": "visual",
            }
        ],
        "limitations": ["string"],
    },
    "aboutness": {
        "summary": "string",
        "subjects": ["string"],
        "themes": ["string"],
        "genres_or_forms": ["string"],
        "evidence": [
            {
                "description": "string",
                "source": "visual|transcript|both",
                "timestamp_seconds": "number|null",
            }
        ],
        "limitations": ["string"],
    },
    "entities": {
        "visible_text": [
            {
                "text": "string",
                "timestamp_seconds": "number|null",
                "source": "visual",
            }
        ],
        "people": [
            {
                "description": "string",
                "timestamp_seconds": "number|null",
                "source": "visual",
            }
        ],
        "objects": [
            {
                "description": "string",
                "timestamp_seconds": "number|null",
                "source": "visual",
            }
        ],
        "limitations": ["string"],
    },
    "activities": {
        "activities_and_events": [
            {
                "description": "string",
                "start_seconds": "number|null",
                "end_seconds": "number|null",
                "source": "visual|transcript|both",
            }
        ],
        "limitations": ["string"],
    },
    "shots": {
        "shots": [
            {
                "start_seconds": "number|null",
                "end_seconds": "number|null",
                "description": "string",
                "transition": "cut|fade|dissolve|unknown|null",
                "source": "visual",
            }
        ],
        "limitations": ["string"],
    },
}

TASK_INSTRUCTIONS = {
    "combined": (
        "Describe overall aboutness and the most useful visible text, people, "
        "objects, activities, events, and shot progression."
    ),
    "aboutness": (
        "Describe whole-video aboutness for archival discovery. Prefer broad, "
        "well-supported subjects and themes over isolated details."
    ),
    "entities": (
        "Extract visible text, non-identifying descriptions of people, and "
        "salient objects. Attach a visual timestamp to each grounded item."
    ),
    "activities": (
        "Extract visually grounded activities and events as time intervals. "
        "Do not turn transcript-only statements into visible events."
    ),
    "shots": (
        "Describe observable shot or scene intervals and transitions. Sparse "
        "sampled frames may not prove exact cuts; state that limitation."
    ),
}


def build_prompt(task: str, transcript: str | None = None) -> str:
    """Build a bounded JSON prompt with explicit evidence-source rules."""

    if task not in TASK_SCHEMAS:
        raise ValueError(f"unknown task: {task}")
    prompt = (
        "Analyze this video for archival discovery and navigation.\n"
        f"{TASK_INSTRUCTIONS[task]}\n\n"
        f"{COMMON_RULES}\n\n"
        "Requested JSON shape:\n"
        f"{json.dumps(TASK_SCHEMAS[task], indent=2)}"
    )
    if transcript is None:
        return prompt + "\n\nNo transcript is supplied; use source \"visual\" only."
    return (
        prompt
        + "\n\nA matching external ASR transcript follows. The model did not hear "
        "the video's audio. Use the transcript only as separately attributed "
        "text context.\n<transcript>\n"
        + transcript
        + "\n</transcript>"
    )


def parse_json_response(text: str) -> tuple[Any | None, str | None]:
    """Parse a JSON object, tolerating a single surrounding Markdown fence."""

    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as error:
        return None, str(error)
    if not isinstance(parsed, dict):
        return None, "response JSON is not an object"
    return parsed, None


def collect_reported_timestamps(value: Any) -> list[float]:
    """Collect finite non-negative timestamp fields from parsed model output."""

    found: set[float] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if key in {"timestamp_seconds", "start_seconds", "end_seconds"}:
                    if (
                        isinstance(nested, (int, float))
                        and not isinstance(nested, bool)
                        and math.isfinite(float(nested))
                        and nested >= 0
                    ):
                        found.add(float(nested))
                else:
                    visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return sorted(found)


def model_timestamp_tokens(
    frame_indices: list[int],
    source_fps: float,
    temporal_patch_size: int,
) -> list[float]:
    """Reproduce Qwen3-VL timestamp grouping for sampled source frames."""

    if source_fps <= 0:
        raise ValueError("source_fps must be positive")
    if temporal_patch_size < 1:
        raise ValueError("temporal_patch_size must be positive")
    if not frame_indices:
        return []
    padded = list(frame_indices)
    remainder = len(padded) % temporal_patch_size
    if remainder:
        padded.extend([padded[-1]] * (temporal_patch_size - remainder))
    return [
        sum(padded[offset : offset + temporal_patch_size])
        / temporal_patch_size
        / source_fps
        for offset in range(0, len(padded), temporal_patch_size)
    ]


def plan_keyframes(
    reported_timestamps: list[float],
    visible_timestamps: list[float],
    *,
    max_keyframes: int,
) -> list[dict[str, float]]:
    """Map reported timestamps to unique nearest model-visible timestamps."""

    if max_keyframes < 0:
        raise ValueError("max_keyframes must be non-negative")
    if not visible_timestamps or max_keyframes == 0:
        return []
    mappings: list[dict[str, float]] = []
    used_visible: set[float] = set()
    for reported in reported_timestamps:
        visible = min(visible_timestamps, key=lambda item: abs(item - reported))
        if visible in used_visible:
            continue
        mappings.append(
            {
                "reported_timestamp_seconds": reported,
                "visible_timestamp_seconds": visible,
                "absolute_difference_seconds": abs(visible - reported),
            }
        )
        used_visible.add(visible)
        if len(mappings) == max_keyframes:
            break
    return mappings


def validate_sampling(sample_fps: float | None, num_frames: int | None) -> None:
    """Require exactly one positive sampling control."""

    if (sample_fps is None) == (num_frames is None):
        raise ValueError("supply exactly one of sample_fps or num_frames")
    if sample_fps is not None and sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if num_frames is not None and num_frames < 2:
        raise ValueError("num_frames must be at least 2")


def bind_video_sampler(
    sample_frames: Any,
    sample_fps: float | None,
    num_frames: int | None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> Any:
    """Bind controls and optionally constrain sampling to an absolute segment."""

    validate_sampling(sample_fps, num_frames)
    if start_seconds is None and end_seconds is None:
        return partial(sample_frames, fps=sample_fps, num_frames=num_frames)
    if start_seconds is None or end_seconds is None:
        raise ValueError("segment sampling requires both start and end seconds")
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("segment times must satisfy 0 <= start < end")

    def sample_segment(metadata: Any, **_: Any) -> list[int]:
        source_fps = float(metadata.fps)
        start_frame = max(0, math.ceil(start_seconds * source_fps))
        end_frame = min(
            int(metadata.total_num_frames) - 1,
            math.floor(end_seconds * source_fps),
        )
        if end_frame <= start_frame:
            raise ValueError("requested segment does not overlap the video")
        segment_metadata = copy(metadata)
        segment_metadata.total_num_frames = end_frame - start_frame + 1
        segment_metadata.duration = segment_metadata.total_num_frames / source_fps
        relative = sample_frames(
            metadata=segment_metadata,
            fps=sample_fps,
            num_frames=num_frames,
        )
        return [start_frame + int(index) for index in relative]

    return sample_segment


def cached_model_revision(config_path: str | Path | None) -> str | None:
    """Extract the immutable Hugging Face snapshot revision from a cache path."""

    if not config_path:
        return None
    path = Path(config_path)
    if path.parent.parent.name != "snapshots":
        return None
    return path.parent.name


def json_safe(value: Any) -> Any:
    """Convert tensors, arrays, paths, and native metadata to JSON data."""

    if value is None:
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
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


def serialize_video_metadata(metadata: Any) -> dict[str, Any]:
    """Project stable public video metadata fields to JSON."""

    names = (
        "total_num_frames",
        "fps",
        "duration",
        "frames_indices",
        "height",
        "width",
        "video_backend",
    )
    return {
        name: json_safe(getattr(metadata, name))
        for name in names
        if getattr(metadata, name, None) is not None
    }


def pop_video_metadata(inputs: Any) -> Any:
    """Remove processor-only metadata before forwarding inputs to generation."""

    metadata = inputs.pop("video_metadata")
    if not metadata:
        raise ValueError("processor returned no video metadata")
    return metadata[0]


def build_messages(video: Any, prompt: str) -> list[dict[str, Any]]:
    """Build chat messages from a plain path or already decoded video."""

    return Qwen3VLTransformersVideo.build_messages(video, prompt)


def extract_keyframes(
    video_path: Path,
    mappings: list[dict[str, float]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Use Decord to save frames nearest timestamped model claims."""

    output: list[dict[str, Any]] = []
    if not mappings:
        return output
    from decord import VideoReader, cpu
    from PIL import Image

    output_dir.mkdir()
    try:
        reader = VideoReader(str(video_path), ctx=cpu(0))
        source_fps = float(reader.get_avg_fps())
        frame_indices = [
            min(len(reader) - 1, round(item["visible_timestamp_seconds"] * source_fps))
            for item in mappings
        ]
        frames = reader.get_batch(frame_indices).asnumpy()
        for sequence, (mapping, frame_index, frame) in enumerate(
            zip(mappings, frame_indices, frames, strict=True),
            start=1,
        ):
            timestamp = mapping["visible_timestamp_seconds"]
            filename = f"keyframe-{sequence:02d}-{timestamp:.3f}s.jpg"
            Image.fromarray(frame).save(output_dir / filename, quality=92)
            output.append(
                {
                    **mapping,
                    "source_frame_index": frame_index,
                    "file": f"keyframes/{filename}",
                }
            )
    except Exception as error:
        return [{"error": f"Decord keyframe extraction failed: {error}"}]
    return output


def package_version(name: str) -> str | None:
    """Return an installed distribution version when available."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args() -> argparse.Namespace:
    """Parse explicit model, input, sampling, prompt, and output arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--task", choices=sorted(TASK_SCHEMAS), default="combined")
    parser.add_argument("--transcript", type=Path)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--sample-fps", type=float)
    sampling.add_argument("--num-frames", type=int)
    parser.add_argument(
        "--max-total-video-pixels",
        type=int,
        default=DEFAULT_MAX_TOTAL_VIDEO_PIXELS,
    )
    parser.add_argument(
        "--min-total-video-pixels",
        type=int,
        default=DEFAULT_MIN_TOTAL_VIDEO_PIXELS,
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--start-seconds", type=float)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument(
        "--video-backend",
        choices=("decord", "torchcodec", "pyav"),
        default="decord",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--max-keyframes", type=int, default=12)
    args = parser.parse_args()
    if args.sample_fps is None and args.num_frames is None:
        args.sample_fps = 0.2
    return args


def main() -> None:
    """Run one real-model probe and retain native output plus run context."""

    args = parse_args()
    validate_sampling(args.sample_fps, args.num_frames)
    if (args.start_seconds is None) != (args.end_seconds is None):
        raise ValueError("supply both start-seconds and end-seconds")
    if args.max_total_video_pixels < args.min_total_video_pixels:
        raise ValueError("max-total-video-pixels must be at least the minimum")
    if not args.video_path.is_file():
        raise FileNotFoundError(args.video_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    import transformers
    import yaml
    from transformers.utils.hub import cached_file
    from transformers.video_utils import load_video

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    transcript = (
        args.transcript.read_text(encoding="utf-8")
        if args.transcript is not None
        else None
    )
    prompt = build_prompt(args.task, transcript)
    started_at = datetime.now().astimezone()
    wall_start = time.perf_counter()
    max_rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)

    load_start = time.perf_counter()
    tool = Qwen3VLTransformersVideo(
        args.model_id,
        revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
    )
    processor = tool.processor
    model = tool.model
    processor.video_processor.size = {
        "longest_edge": args.max_total_video_pixels,
        "shortest_edge": args.min_total_video_pixels,
    }
    model_load_seconds = time.perf_counter() - load_start

    decode_start = time.perf_counter()
    decoded_video, decoded_metadata = load_video(
        str(args.video_path.resolve()),
        backend=args.video_backend,
        sample_indices_fn=bind_video_sampler(
            processor.video_processor.sample_frames,
            args.sample_fps,
            args.num_frames,
            args.start_seconds,
            args.end_seconds,
        ),
    )
    decode_seconds = time.perf_counter() - decode_start
    messages = build_messages(decoded_video, prompt)

    preprocess_start = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "video_metadata": [decoded_metadata],
            "do_sample_frames": False,
            "return_metadata": True,
            "cap_pixels_per_frame": True,
        },
    )
    preprocess_seconds = time.perf_counter() - preprocess_start
    metadata = pop_video_metadata(inputs)
    video_metadata = serialize_video_metadata(metadata)
    source_fps = float(video_metadata["fps"])
    frame_indices = [int(item) for item in video_metadata["frames_indices"]]
    temporal_patch_size = int(processor.video_processor.temporal_patch_size)
    visible_timestamps = model_timestamp_tokens(
        frame_indices,
        source_fps,
        temporal_patch_size,
    )
    input_tokens = int(inputs["input_ids"].shape[-1])
    video_grid_thw = json_safe(inputs.get("video_grid_thw"))
    grid = video_grid_thw[0]
    visual_tokens = math.prod(grid) // int(processor.video_processor.merge_size) ** 2
    generation_start = time.perf_counter()
    generated_ids = tool.generate(
        inputs,
        generation_options={"max_new_tokens": args.max_new_tokens},
    )
    torch.cuda.synchronize(args.device)
    generation_seconds = time.perf_counter() - generation_start
    trimmed = generated_ids[:, input_tokens:]
    response_text = tool.decode(generated_ids, input_tokens)
    output_tokens = int(trimmed.shape[-1])
    structured, parse_error = parse_json_response(response_text)
    reported_timestamps = (
        collect_reported_timestamps(structured) if structured is not None else []
    )
    keyframe_plan = plan_keyframes(
        reported_timestamps,
        visible_timestamps,
        max_keyframes=args.max_keyframes,
    )
    keyframes = extract_keyframes(
        args.video_path,
        keyframe_plan,
        args.output_dir / "keyframes",
    )

    completed_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - wall_start
    result = {
        "native_type": "transformers.GenerationMixin.generate text decode",
        "response_text": response_text,
        "structured": structured,
        "parse_error": parse_error,
    }
    manifest = {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "model_load_seconds": model_load_seconds,
        "decode_seconds": decode_seconds,
        "preprocess_seconds": preprocess_seconds,
        "generation_seconds": generation_seconds,
        "model_id": args.model_id,
        "requested_model_revision": args.model_revision,
        "resolved_model_revision": cached_model_revision(
            cached_file(
                args.model_id,
                "config.json",
                revision=args.model_revision,
            )
        ),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": package_version("accelerate"),
            "torchcodec": package_version("torchcodec"),
            "decord": package_version("decord"),
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name(args.device),
            "cuda_runtime": torch.version.cuda,
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "attention_implementation": args.attention_implementation,
            "dtype": args.dtype,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "visual_tokens": visual_tokens,
        },
        "video": {
            "sampling_request": {
                "sample_fps": args.sample_fps,
                "num_frames": args.num_frames,
                "max_total_video_pixels": args.max_total_video_pixels,
                "min_total_video_pixels": args.min_total_video_pixels,
                "video_backend": args.video_backend,
                "cap_pixels_per_frame": True,
                "start_seconds": args.start_seconds,
                "end_seconds": args.end_seconds,
            },
            "processor_metadata": video_metadata,
            "video_grid_thw": video_grid_thw,
            "effective_sample_count": len(frame_indices),
            "model_timestamp_count": len(visible_timestamps),
            "temporal_patch_size": temporal_patch_size,
            "model_timestamp_tokens_seconds": visible_timestamps,
        },
        "prompt_task": args.task,
        "transcript_supplied": args.transcript is not None,
        "resource": {
            "max_rss_before_kib": max_rss_before_kib,
            "max_rss_after_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
        },
        "reported_timestamps_seconds": reported_timestamps,
        "keyframes": keyframes,
    }

    (args.output_dir / "native_output.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(json_safe(manifest), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (args.output_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "input_ref.txt").write_text(
        f"video: {args.video_path.resolve()}\n"
        f"transcript: {args.transcript.resolve() if args.transcript else 'none'}\n"
        "cleanup: caller-owned inputs were not modified or deleted\n",
        encoding="utf-8",
    )
    (args.output_dir / "observations.md").write_text(
        "# Observations\n\n"
        "Review `native_output.json` for schema validity, grounded coverage, "
        "contradictions, source attribution, and temporal accuracy. Keyframes "
        "show the nearest model-visible timestamp, not proof that the reported "
        "event spans the claimed interval.\n\n"
        f"- Task: `{args.task}`\n"
        f"- Transcript supplied: {args.transcript is not None}\n"
        f"- Input tokens: {input_tokens}\n"
        f"- Output tokens: {output_tokens}\n"
        f"- JSON parse error: {parse_error or 'none'}\n"
        f"- Wall time: {elapsed_seconds:.3f} seconds\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
