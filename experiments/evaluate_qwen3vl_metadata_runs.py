#!/usr/bin/env python3
"""Summarize Qwen3-VL metadata runs without claiming automatic accuracy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any, Iterator

import yaml


TIMESTAMP_FIELDS = {"timestamp_seconds", "start_seconds", "end_seconds"}


def walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    """Yield every dictionary in a nested JSON-compatible value."""

    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from walk_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from walk_dicts(nested)


def array_counts(value: Any, prefix: str = "") -> dict[str, int]:
    """Count arrays by dotted response path as a coverage diagnostic."""

    counts: dict[str, int] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(nested, list):
                counts[path] = len(nested)
            counts.update(array_counts(nested, path))
    elif isinstance(value, list):
        for nested in value:
            counts.update(array_counts(nested, prefix))
    return counts


def is_metadata_item(value: dict[str, Any]) -> bool:
    """Return whether a dictionary represents an attributable metadata item."""

    return bool(TIMESTAMP_FIELDS.intersection(value)) and (
        "description" in value or "text" in value
    )


def finite_timestamp(value: Any) -> float | None:
    """Convert a finite non-negative numeric timestamp, excluding booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    timestamp = float(value)
    if timestamp < 0 or not math.isfinite(timestamp):
        return None
    return timestamp


def analyze_run(
    name: str,
    manifest: dict[str, Any],
    native_output: dict[str, Any],
) -> dict[str, Any]:
    """Calculate structural, attribution, temporal, and resource diagnostics."""

    structured = native_output.get("structured")
    dictionaries = list(walk_dicts(structured)) if structured is not None else []
    metadata_items = [item for item in dictionaries if is_metadata_item(item)]
    source_counts: dict[str, int] = {}
    missing_source = 0
    interval_inversions = 0
    timestamps: list[float] = []
    for item in metadata_items:
        source = item.get("source")
        if source is None:
            missing_source += 1
        else:
            source_counts[str(source)] = source_counts.get(str(source), 0) + 1
        for key in TIMESTAMP_FIELDS:
            timestamp = finite_timestamp(item.get(key))
            if timestamp is not None:
                timestamps.append(timestamp)
        start = finite_timestamp(item.get("start_seconds"))
        end = finite_timestamp(item.get("end_seconds"))
        if start is not None and end is not None and end < start:
            interval_inversions += 1

    video = manifest["video"]
    request = video["sampling_request"]
    start_bound = request.get("start_seconds")
    end_bound = request.get("end_seconds")
    if start_bound is None:
        start_bound = 0.0
    if end_bound is None:
        end_bound = video["processor_metadata"].get("duration")
    outside_segment = [
        timestamp
        for timestamp in timestamps
        if timestamp < float(start_bound)
        or (end_bound is not None and timestamp > float(end_bound))
    ]

    visible = [float(item) for item in video["model_timestamp_tokens_seconds"]]
    nearest_differences = [
        min(abs(timestamp - candidate) for candidate in visible)
        for timestamp in timestamps
        if visible
    ]
    generation = manifest["generation"]
    resource = manifest["resource"]
    response_text = str(native_output.get("response_text", ""))
    transcript_supplied = bool(manifest["transcript_supplied"])
    transcript_attributions = source_counts.get("transcript", 0) + source_counts.get(
        "both", 0
    )

    return {
        "run": name,
        "task": manifest["prompt_task"],
        "transcript_supplied": transcript_supplied,
        "json_valid": structured is not None and native_output.get("parse_error") is None,
        "array_item_counts": array_counts(structured),
        "metadata_item_count": len(metadata_items),
        "source_counts": source_counts,
        "missing_source_count": missing_source,
        "transcript_attribution_count": transcript_attributions,
        "transcript_denial_detected": transcript_supplied
        and re.search(r"\bno\b.{0,24}\btranscript\b", response_text.lower())
        is not None,
        "reported_timestamp_count": len(timestamps),
        "timestamps_outside_requested_segment_count": len(outside_segment),
        "timestamps_outside_requested_segment": outside_segment,
        "interval_inversion_count": interval_inversions,
        "nearest_visible_timestamp_difference_seconds": {
            "mean": mean(nearest_differences) if nearest_differences else None,
            "max": max(nearest_differences) if nearest_differences else None,
        },
        "sampling": {
            **request,
            "effective_sample_count": video.get(
                "effective_sample_count",
                len(video["processor_metadata"]["frames_indices"]),
            ),
            "model_timestamp_count": video.get(
                "model_timestamp_count",
                len(visible),
            ),
        },
        "tokens": {
            "input": generation["input_tokens"],
            "output": generation["output_tokens"],
            "visual": generation.get("visual_tokens"),
        },
        "seconds": {
            "elapsed": manifest["elapsed_seconds"],
            "load": manifest["model_load_seconds"],
            "decode": manifest["decode_seconds"],
            "preprocess": manifest["preprocess_seconds"],
            "generation": manifest["generation_seconds"],
        },
        "cuda_peak_allocated_bytes": resource["cuda_peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": resource["cuda_peak_reserved_bytes"],
    }


def compare_runs(
    baseline: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Calculate transparent deltas between two diagnostic summaries."""

    all_sections = sorted(
        set(baseline["array_item_counts"]) | set(comparison["array_item_counts"])
    )
    return {
        "baseline": baseline["run"],
        "comparison": comparison["run"],
        "delta": {
            "input_tokens": comparison["tokens"]["input"]
            - baseline["tokens"]["input"],
            "output_tokens": comparison["tokens"]["output"]
            - baseline["tokens"]["output"],
            "generation_seconds": comparison["seconds"]["generation"]
            - baseline["seconds"]["generation"],
            "cuda_peak_allocated_bytes": comparison["cuda_peak_allocated_bytes"]
            - baseline["cuda_peak_allocated_bytes"],
            "metadata_item_count": comparison["metadata_item_count"]
            - baseline["metadata_item_count"],
            "array_item_counts": {
                section: comparison["array_item_counts"].get(section, 0)
                - baseline["array_item_counts"].get(section, 0)
                for section in all_sections
            },
        },
    }


def load_run(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one retained run's manifest and native output."""

    manifest = yaml.safe_load((path / "manifest.yaml").read_text(encoding="utf-8"))
    native = json.loads((path / "native_output.json").read_text(encoding="utf-8"))
    return manifest, native


def parse_args() -> argparse.Namespace:
    """Parse retained run directories and optional named pair comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        default=[],
        metavar=("BASELINE", "COMPARISON"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write reviewable diagnostic JSON for the selected retained runs."""

    args = parse_args()
    analyses: dict[str, dict[str, Any]] = {}
    for run_dir in args.run_dirs:
        manifest, native = load_run(run_dir)
        analyses[run_dir.name] = analyze_run(run_dir.name, manifest, native)

    output: dict[str, Any] = {
        "method": (
            "Structural and resource diagnostics only. Coverage counts are not "
            "accuracy, and correctness/contradictions require direct review."
        ),
        "runs": analyses,
    }
    if args.pair:
        output["pairs"] = [
            compare_runs(analyses[baseline_name], analyses[comparison_name])
            for baseline_name, comparison_name in args.pair
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
