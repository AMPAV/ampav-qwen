"""Native Transformers wrapper for Qwen3-Omni audiovisual generation."""

from __future__ import annotations

from contextlib import nullcontext
from os import PathLike
from pathlib import Path
from typing import Any, Callable

from .vl.transformers import _input_token_count, _trim_sequences


DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


class Qwen3OmniTransformers:
    """Generate text from Qwen3-Omni text, audio, image, or video inputs.

    The wrapper loads only the native thinker component because AMPAV metadata
    generation needs text output, not the model's speech-generation talker.
    ``use_audio_in_video`` explicitly controls whether a video's audio track is
    provided to the model; it never changes implicitly based on the input.
    """

    def __init__(
        self,
        model_id: str | PathLike[str] = DEFAULT_MODEL_ID,
        *,
        revision: str | None = None,
        device_map: str | dict[str, Any] = "auto",
        dtype: str = "auto",
        attention_implementation: str = "sdpa",
        model_load_options: dict[str, Any] | None = None,
        model: Any | None = None,
        processor: Any | None = None,
        process_mm_info: Callable[..., Any] | None = None,
    ) -> None:
        supplied = (model is not None, processor is not None, process_mm_info is not None)
        if any(supplied) and not all(supplied):
            raise ValueError(
                "model, processor, and process_mm_info must be supplied together"
            )

        self.model_id = str(model_id)
        self.revision = revision
        self.device_map = device_map
        self.dtype = dtype
        self.attention_implementation = attention_implementation
        self._torch: Any | None = None

        if model is None:
            try:
                import torch
                from qwen_omni_utils import process_mm_info as native_process_mm_info
                from transformers import (
                    Qwen3OmniMoeProcessor,
                    Qwen3OmniMoeThinkerForConditionalGeneration,
                )
            except ImportError as error:
                raise ImportError(
                    "Qwen3-Omni media support requires ampav-qwen[omni]"
                ) from error

            load_options = {
                "revision": revision,
                "device_map": device_map,
                "dtype": dtype,
                "attn_implementation": attention_implementation,
                **(model_load_options or {}),
            }
            model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                self.model_id,
                **load_options,
            )
            model.eval()
            processor = Qwen3OmniMoeProcessor.from_pretrained(
                self.model_id,
                revision=revision,
            )
            process_mm_info = native_process_mm_info
            self._torch = torch

        self.model = model
        self.processor = processor
        self._process_mm_info = process_mm_info

    @property
    def input_device(self) -> Any:
        """Return the native model device receiving prepared inputs."""

        return self.model.device

    @staticmethod
    def build_messages(
        video: str | PathLike[str],
        prompt: str,
        *,
        video_options: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Build one audiovisual chat message with explicit video controls."""

        video_path = str(Path(video).resolve())
        video_content = {
            "type": "video",
            "video": video_path,
            **(video_options or {}),
        }
        return [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def prepare(
        self,
        video: str | PathLike[str],
        prompt: str,
        *,
        use_audio_in_video: bool,
        video_options: dict[str, Any] | None = None,
        processor_options: dict[str, Any] | None = None,
    ) -> Any:
        """Decode media and return native Qwen3-Omni processor inputs."""

        messages = self.build_messages(
            video,
            prompt,
            video_options=video_options,
        )
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = self._process_mm_info(
            messages,
            use_audio_in_video=use_audio_in_video,
        )
        options = {
            "text": text,
            "audio": audios,
            "images": images,
            "videos": videos,
            "return_tensors": "pt",
            "padding": True,
            "use_audio_in_video": use_audio_in_video,
            **(processor_options or {}),
        }
        return self.processor(**options)

    def generate(
        self,
        inputs: Any,
        *,
        use_audio_in_video: bool,
        generation_options: dict[str, Any] | None = None,
    ) -> Any:
        """Generate and return native thinker token IDs."""

        model_inputs = inputs
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(self.input_device)
            model_dtype = getattr(self.model, "dtype", None)
            if model_dtype is not None:
                model_inputs = model_inputs.to(model_dtype)
        options = {
            "use_audio_in_video": use_audio_in_video,
            "do_sample": False,
            **(generation_options or {}),
        }
        context = self._torch.inference_mode() if self._torch is not None else nullcontext()
        with context:
            return self.model.generate(**model_inputs, **options)

    def decode(self, generated_ids: Any, input_tokens: int) -> str:
        """Decode only tokens generated after the input prompt."""

        sequences = getattr(generated_ids, "sequences", generated_ids)
        trimmed = _trim_sequences(sequences, input_tokens)
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def process(
        self,
        video: str | PathLike[str],
        prompt: str,
        *,
        use_audio_in_video: bool,
        video_options: dict[str, Any] | None = None,
        processor_options: dict[str, Any] | None = None,
        generation_options: dict[str, Any] | None = None,
    ) -> str:
        """Prepare, generate, and decode one explicit modality request."""

        inputs = self.prepare(
            video,
            prompt,
            use_audio_in_video=use_audio_in_video,
            video_options=video_options,
            processor_options=processor_options,
        )
        generated_ids = self.generate(
            inputs,
            use_audio_in_video=use_audio_in_video,
            generation_options=generation_options,
        )
        return self.decode(generated_ids, _input_token_count(inputs))
