"""Native Transformers wrapper for Qwen3-VL video generation."""

from __future__ import annotations

from contextlib import nullcontext
from os import PathLike
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def _input_token_count(inputs: Any) -> int:
    input_ids = inputs["input_ids"]
    shape = getattr(input_ids, "shape", None)
    return int(shape[-1]) if shape is not None else len(input_ids[0])


def _trim_sequences(sequences: Any, input_tokens: int) -> Any:
    try:
        return sequences[:, input_tokens:]
    except TypeError:
        return [sequence[input_tokens:] for sequence in sequences]


class Qwen3VLTransformersVideo:
    """Run Qwen3-VL video-to-text generation with native Transformers.

    Args:
        model_id: Hugging Face model ID or local model path.
        revision: Optional immutable model revision.
        device: Torch device receiving the model and prepared inputs.
        dtype: Torch dtype name used while loading the model.
        attention_implementation: Native Transformers attention backend.
        model: Optional injected native model for testing or managed loading.
        processor: Optional injected native processor paired with ``model``.

    ``generate`` returns native generated token IDs. ``process`` performs the
    ordinary chat preparation and returns decoded text. Callers retain control
    over processor and generation options.
    """

    def __init__(
        self,
        model_id: str | PathLike[str] = DEFAULT_MODEL_ID,
        *,
        revision: str | None = None,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be supplied together")

        self.model_id = str(model_id)
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.attention_implementation = attention_implementation
        self._torch: Any | None = None

        if model is None:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            native_dtype = getattr(torch, dtype)
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                revision=revision,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                revision=revision,
                dtype=native_dtype,
                attn_implementation=attention_implementation,
            ).to(device)
            model.eval()
            self._torch = torch

        self.processor = processor
        self.model = model

    @staticmethod
    def build_messages(video: Any, prompt: str) -> list[dict[str, Any]]:
        """Build one native video chat message from a path or decoded input."""

        if isinstance(video, (Path, PathLike)):
            video = str(Path(video).resolve())
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def prepare(
        self,
        video: Any,
        prompt: str,
        *,
        processor_options: dict[str, Any] | None = None,
    ) -> Any:
        """Prepare native model inputs while preserving caller options."""

        options = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            **(processor_options or {}),
        }
        return self.processor.apply_chat_template(
            self.build_messages(video, prompt),
            **options,
        )

    def generate(
        self,
        inputs: Any,
        *,
        generation_options: dict[str, Any] | None = None,
    ) -> Any:
        """Generate and return native Transformers token IDs."""

        model_inputs = inputs.to(self.device) if hasattr(inputs, "to") else inputs
        options = {"do_sample": False, **(generation_options or {})}
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
        video: Any,
        prompt: str,
        *,
        processor_options: dict[str, Any] | None = None,
        generation_options: dict[str, Any] | None = None,
    ) -> str:
        """Prepare, generate, and decode one video request."""

        inputs = self.prepare(video, prompt, processor_options=processor_options)
        generated_ids = self.generate(inputs, generation_options=generation_options)
        return self.decode(generated_ids, _input_token_count(inputs))
