"""OpenAI-compatible vLLM client for Qwen3-Omni text output."""

from __future__ import annotations

import base64
from os import PathLike
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs and generating "
    "text and speech."
)


def _video_url(video: str | PathLike[str]) -> str:
    value = str(video)
    if value.startswith(("http://", "https://", "data:")):
        return value

    path = Path(video)
    if not path.is_file():
        raise FileNotFoundError(path)
    mime_type = {
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".mov": "video/quicktime",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }.get(path.suffix.lower(), "video/mp4")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class Qwen3OmniVllm:
    """Call Qwen3-Omni through a regular vLLM or vLLM-Omni server.

    Both runtimes expose the same OpenAI-compatible chat-completions boundary.
    ``use_audio_in_video`` explicitly asks the native processor to extract and
    use the video's audio track. Output modalities are restricted to text.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_VLLM_MODEL_ID,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        timeout: float = 1800.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError(
                    "vLLM client support requires the optional OpenAI package"
                ) from error
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )
        self.client = client
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def build_messages(
        prompt: str,
        *,
        video: str | PathLike[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the documented Qwen3-Omni OpenAI message shape."""

        content: list[dict[str, Any]] = []
        if video is not None:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": _video_url(video)},
                }
            )
        content.append({"type": "text", "text": prompt})
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {"role": "user", "content": content},
        ]

    def process(
        self,
        prompt: str,
        *,
        video: str | PathLike[str] | None = None,
        use_audio_in_video: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = False,
        request_options: dict[str, Any] | None = None,
    ) -> Any:
        """Return the native OpenAI completion or completion stream."""

        if use_audio_in_video and video is None:
            raise ValueError("use_audio_in_video requires a video input")
        extra_body: dict[str, Any] = {}
        if use_audio_in_video:
            extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}
        options = {
            "model": self.model_id,
            "messages": self.build_messages(prompt, video=video),
            "modalities": ["text"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "extra_body": extra_body,
            **(request_options or {}),
        }
        return self.client.chat.completions.create(**options)
