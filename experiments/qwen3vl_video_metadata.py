#!/usr/bin/env python3
"""Run a retained Qwen3-VL video-metadata probe with OpenVINO GenAI."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import resource
import shlex
import sys
import time
from typing import Any

import av
import numpy as np
from openvino import Core, Tensor, __version__ as openvino_version
import openvino_genai
import yaml

from ampav.qwen3vl import Qwen3VLVideo


DEFAULT_PROMPT = """Analyze this video for archival discovery and navigation.
Report only observations grounded in the supplied video. Use concise plain-text
headings for visible text, people without identity assumptions, objects,
activities and events, setting, scene progression, significant moments,
overall subject, themes, and evidence limitations. Include timestamps only
when the visual evidence supports them, and label uncertainty explicitly."""


def select_frame_indices(
    total_frames: int,
    source_fps: float,
    sample_fps: float,
    max_frames: int | None,
) -> list[int]:
    """Select approximately uniform source-frame indices across the video."""

    if total_frames < 1:
        raise ValueError("video reports no frames")
    if source_fps <= 0:
        raise ValueError("video reports no frame rate")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    step = max(1, round(source_fps / sample_fps))
    candidates = list(range(0, total_frames, step))
    if max_frames is None or len(candidates) <= max_frames:
        return candidates
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2 when supplied")
    positions = np.linspace(0, len(candidates) - 1, max_frames, dtype=int)
    return [candidates[position] for position in positions]


def scaled_dimensions(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Return even dimensions within a pixel budget without upscaling."""

    if width < 1 or height < 1:
        raise ValueError("video dimensions must be positive")
    if max_pixels < 4:
        raise ValueError("max_pixels must be at least 4")
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    scaled_width = max(2, int(width * scale) // 2 * 2)
    scaled_height = max(2, int(height * scale) // 2 * 2)
    return scaled_width, scaled_height


def read_sampled_video(
    path: Path,
    sample_fps: float,
    max_frames: int | None,
    max_frame_pixels: int,
) -> tuple[Tensor, openvino_genai.VideoMetadata, dict[str, Any]]:
    """Decode selected frames and describe the effective native video tensor."""

    with av.open(path) as container:
        stream = container.streams.video[0]
        source_frame_count = stream.frames
        source_width = stream.width
        source_height = stream.height
        source_fps = float(stream.average_rate) if stream.average_rate else 0.0
        indices = select_frame_indices(
            source_frame_count,
            source_fps,
            sample_fps,
            max_frames,
        )
        target_width, target_height = scaled_dimensions(
            source_width,
            source_height,
            max_frame_pixels,
        )
        frames: list[np.ndarray[Any, Any]] = []
        timestamps: list[float] = []
        target_position = 0
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index != indices[target_position]:
                continue
            frames.append(
                frame.reformat(
                    width=target_width,
                    height=target_height,
                    format="bgr24",
                ).to_ndarray()
            )
            timestamps.append(
                float(frame.time)
                if frame.time is not None
                else frame_index / source_fps
            )
            target_position += 1
            if target_position == len(indices):
                break

    if len(frames) != len(indices):
        raise RuntimeError(
            f"decoded {len(frames)} selected frames; expected {len(indices)}"
        )
    effective_fps = (
        (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        if len(timestamps) > 1 and timestamps[-1] > timestamps[0]
        else sample_fps
    )
    metadata = openvino_genai.VideoMetadata()
    metadata.fps = effective_fps
    video_array = np.stack(frames)
    return (
        Tensor(video_array),
        metadata,
        {
            "source_fps": source_fps,
            "source_frame_count": source_frame_count,
            "selected_frame_indices": indices,
            "selected_frame_timestamps_seconds": timestamps,
            "effective_sample_fps": effective_fps,
            "source_width": source_width,
            "source_height": source_height,
            "max_frame_pixels": max_frame_pixels,
            "tensor_shape": list(video_array.shape),
        },
    )


def metric_value(value: Any) -> Any:
    """Convert a native metric scalar or mean/std pair to JSON data."""

    if hasattr(value, "mean") and hasattr(value, "std"):
        return {"mean": float(value.mean), "std": float(value.std)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def serialize_metrics(metrics: Any) -> dict[str, Any]:
    """Read the public OpenVINO VLM performance metrics without relabeling."""

    method_names = [
        "get_load_time",
        "get_num_input_tokens",
        "get_num_generated_tokens",
        "get_generate_duration",
        "get_inference_duration",
        "get_tokenization_duration",
        "get_detokenization_duration",
        "get_sampling_duration",
        "get_prepare_embeddings_duration",
        "get_vision_encoding_duration",
        "get_text_embedding_duration",
        "get_ttft",
        "get_tpot",
        "get_throughput",
        "get_total_image_slice_count",
    ]
    output: dict[str, Any] = {}
    for name in method_names:
        method = getattr(metrics, name, None)
        if method is None:
            continue
        try:
            output[name] = metric_value(method())
        except RuntimeError as error:
            output[name] = {"error": str(error)}
    return output


def serialize_result(result: openvino_genai.VLMDecodedResults) -> dict[str, Any]:
    """Create a safe readable projection of the native decoded result."""

    return {
        "native_type": f"{type(result).__module__}.{type(result).__name__}",
        "texts": list(result.texts),
        "scores": [float(score) for score in result.scores],
        "finish_reasons": [str(reason) for reason in result.finish_reasons],
        "perf_metrics": serialize_metrics(result.perf_metrics),
    }


def build_prompt(prompt: str, transcript_path: Path | None) -> str:
    """Append an explicitly identified matching transcript when supplied."""

    if transcript_path is None:
        return prompt
    transcript = transcript_path.read_text(encoding="utf-8")
    return (
        f"{prompt}\n\nA matching ASR transcript is provided as context. "
        "Use it only to clarify the supplied video and distinguish transcript "
        "evidence from visible evidence.\n\n<transcript>\n"
        f"{transcript}\n</transcript>"
    )


def parse_args() -> argparse.Namespace:
    """Parse explicit model, input, sampling, and output arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-id", default="OpenVINO/Qwen3-VL-4B-Instruct-int4-ov")
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-frame-pixels", type=int, default=262144)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--transcript", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the real model once and retain native output plus run context."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now().astimezone()
    start = time.perf_counter()
    max_rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    video, video_metadata, video_info = read_sampled_video(
        args.video_path,
        args.sample_fps,
        args.max_frames,
        args.max_frame_pixels,
    )
    prompt = build_prompt(args.prompt, args.transcript)
    load_start = time.perf_counter()
    tool = Qwen3VLVideo(args.model_path, args.device)
    pipeline_load_seconds = time.perf_counter() - load_start
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.do_sample = False

    result = tool.process(
        prompt,
        [video],
        videos_metadata=[video_metadata],
        generation_config=config,
    )
    completed_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - start
    max_rss_after_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    native_output = serialize_result(result)

    manifest = {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "pipeline_load_seconds": pipeline_load_seconds,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "openvino_version": openvino_version,
        "openvino_genai_version": openvino_genai.__version__,
        "device": args.device,
        "available_openvino_devices": Core().available_devices,
        "runtime": platform.platform(),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "video": video_info,
        "transcript_supplied": args.transcript is not None,
        "max_rss_before_kib": max_rss_before_kib,
        "max_rss_after_kib": max_rss_after_kib,
    }
    (args.output_dir / "native_output.json").write_text(
        json.dumps(native_output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "input_ref.txt").write_text(
        f"video: {args.video_path.resolve()}\n"
        f"model: {args.model_path.resolve()}\n"
        f"transcript: {args.transcript.resolve() if args.transcript else 'none'}\n"
        "cleanup: caller-owned inputs were not modified or deleted\n",
        encoding="utf-8",
    )
    (args.output_dir / "observations.md").write_text(
        "# Observations\n\n"
        "Review `native_output.json` for qualitative grounding, omissions, "
        "unsupported inference, temporal accuracy, and discovery value.\n\n"
        f"- Device: `{args.device}`\n"
        f"- Selected frames: {len(video_info['selected_frame_indices'])}\n"
        f"- Transcript supplied: {args.transcript is not None}\n"
        f"- Wall time: {elapsed_seconds:.3f} seconds\n"
        f"- Native finish reasons: {native_output['finish_reasons']}\n",
        encoding="utf-8",
    )
    print(json.dumps(native_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
