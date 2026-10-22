#!/usr/bin/env python3
"""Probe streamed Qwen3-Omni metadata through vLLM-compatible servers."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Iterable
from urllib.request import urlopen

from ampav.qwen3omni import (
    DEFAULT_BASE_URL,
    DEFAULT_VLLM_MODEL_ID,
    Qwen3OmniVllm,
)
from experiments.qwen3_omni_audiovisual_metadata import DEFAULT_PROMPT

MODES = {
    "text": None,
    "video-only": False,
    "audiovisual": True,
}
TEXT_SMOKE_PROMPT = "Reply with exactly this JSON object: {\"status\":\"ok\"}"


def package_version(name: str) -> str | None:
    """Return an installed distribution version when available."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_safe(value: Any) -> Any:
    """Convert native SDK values to reviewable JSON data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    return str(value)


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse one JSON object, tolerating one surrounding Markdown fence."""

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


def parse_gpu_rows(output: str) -> list[dict[str, Any]]:
    """Parse the bounded nvidia-smi CSV used by the experiment."""

    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, name, used, total, utilization = [
            item.strip() for item in line.split(",", 4)
        ]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_percent": int(utilization),
            }
        )
    return rows


def query_gpus() -> list[dict[str, Any]]:
    """Return current GPU memory and utilization from nvidia-smi."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_rows(completed.stdout)


class GpuSampler:
    """Sample server-host GPU state and retain per-device peaks."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start background sampling."""

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for the final bounded poll."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 2)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(
                    {
                        "monotonic_seconds": time.perf_counter(),
                        "gpus": query_gpus(),
                    }
                )
            except (FileNotFoundError, subprocess.SubprocessError, ValueError):
                return
            self._stop.wait(self.interval_seconds)

    def summary(self) -> list[dict[str, Any]]:
        """Return per-device memory and utilization peaks."""

        peaks: dict[int, dict[str, Any]] = {}
        for sample in self.samples:
            for row in sample["gpus"]:
                current = peaks.setdefault(
                    row["index"],
                    {
                        "index": row["index"],
                        "name": row["name"],
                        "memory_total_mib": row["memory_total_mib"],
                        "peak_memory_used_mib": 0,
                        "peak_utilization_percent": 0,
                    },
                )
                current["peak_memory_used_mib"] = max(
                    current["peak_memory_used_mib"], row["memory_used_mib"]
                )
                current["peak_utilization_percent"] = max(
                    current["peak_utilization_percent"],
                    row["utilization_percent"],
                )
        return [peaks[index] for index in sorted(peaks)]


def collect_stream(
    chunks: Iterable[Any],
    *,
    started_at: float,
    clock=time.perf_counter,
) -> dict[str, Any]:
    """Collect streamed native chunks and calculate first-token latency."""

    text_parts: list[str] = []
    native_chunks: list[Any] = []
    first_token_seconds = None
    usage = None
    for chunk in chunks:
        native_chunks.append(json_safe(chunk))
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = json_safe(chunk_usage)
        for choice in getattr(chunk, "choices", []):
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                if first_token_seconds is None:
                    first_token_seconds = clock() - started_at
                text_parts.append(content)
    return {
        "response_text": "".join(text_parts),
        "time_to_first_token_seconds": first_token_seconds,
        "usage": usage,
        "native_chunks": native_chunks,
    }


def fetch_server_json(base_url: str, suffix: str) -> Any:
    """Fetch a small local server metadata endpoint when available."""

    root = base_url.removesuffix("/v1")
    with urlopen(f"{root}{suffix}", timeout=10) as response:
        return json.load(response)


def parse_args() -> argparse.Namespace:
    """Parse bounded runtime, request, and evidence options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--runtime-label", choices=("vllm", "vllm-omni"), required=True
    )
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-id", default=DEFAULT_VLLM_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODES),
        default=list(MODES),
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--gpu-sample-seconds", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Run text and controlled media requests against one selected runtime."""

    args = parse_args()
    media_modes = [mode for mode in args.modes if mode != "text"]
    if media_modes and (args.video_path is None or not args.video_path.is_file()):
        raise FileNotFoundError(args.video_path)
    if args.max_tokens < 1:
        raise ValueError("max-tokens must be positive")
    if args.gpu_sample_seconds <= 0:
        raise ValueError("gpu-sample-seconds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

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
        f"video: {args.video_path.resolve() if args.video_path else 'none'}\n"
        "cleanup: caller-owned input was not modified or deleted\n",
        encoding="utf-8",
    )

    tool = Qwen3OmniVllm(
        args.model_id,
        base_url=args.base_url,
        timeout=args.timeout_seconds,
    )
    results: dict[str, Any] = {}
    started = datetime.now().astimezone()
    for mode in args.modes:
        use_audio = MODES[mode]
        mode_prompt = TEXT_SMOKE_PROMPT if mode == "text" else prompt
        video = args.video_path if mode != "text" else None
        sampler = GpuSampler(args.gpu_sample_seconds)
        sampler.start()
        request_start = time.perf_counter()
        try:
            chunks = tool.process(
                mode_prompt,
                video=video,
                use_audio_in_video=bool(use_audio),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=True,
                request_options={"stream_options": {"include_usage": True}},
            )
            collected = collect_stream(chunks, started_at=request_start)
            total_seconds = time.perf_counter() - request_start
        finally:
            sampler.stop()

        structured, parse_error = parse_json_response(collected["response_text"])
        usage = collected["usage"] or {}
        results[mode] = {
            **collected,
            "structured": structured,
            "parse_error": parse_error,
            "output_valid": structured is not None,
            "use_audio_in_video": use_audio,
            "total_seconds": total_seconds,
            "output_tokens": usage.get("completion_tokens"),
            "gpu_peaks": sampler.summary(),
        }
        (args.output_dir / "native_output.partial.json").write_text(
            json.dumps(
                {"completed_modes": list(results), "results": results},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    server_metadata = {}
    for name, suffix in (("version", "/version"), ("models", "/v1/models")):
        try:
            server_metadata[name] = fetch_server_json(args.base_url, suffix)
        except Exception as error:  # Metadata failure must not discard a valid run.
            server_metadata[name] = {"error": str(error)}

    completed = datetime.now().astimezone()
    (args.output_dir / "native_output.json").write_text(
        json.dumps({"results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    import yaml

    manifest = {
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "runtime_label": args.runtime_label,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "base_url": args.base_url,
        "client": {
            "python": platform.python_version(),
            "ampav_qwen": package_version("ampav-qwen"),
            "openai": package_version("openai"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "vllm": package_version("vllm"),
            "vllm_omni": package_version("vllm-omni"),
        },
        "server": server_metadata,
        "request": {
            "modes": args.modes,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "output_modalities": ["text"],
        },
    }
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    validity = ", ".join(
        f"{mode}={'valid' if result['output_valid'] else 'invalid'}"
        for mode, result in results.items()
    )
    (args.output_dir / "observations.md").write_text(
        "# Observations\n\n"
        f"Initial structural validity: {validity}.\n\n"
        "Review grounded visual, audible, and cross-modal evidence separately. "
        "Classify invalid output as a runtime or invocation result before "
        "making any model-quality judgment.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
