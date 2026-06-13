"""Thin client for NVIDIA NIM's OpenAI-compatible API.

Wraps chat (text), vision (image + text), and embedding calls against
https://integrate.api.nvidia.com/v1 using the official ``openai`` SDK.
Model names are parameterized via environment variables because the NIM
catalog (https://build.nvidia.com/models) changes frequently.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import random
import re
import time
from pathlib import Path
from typing import Callable, TypeVar

import openai
from openai import OpenAI

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_VISION_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
DEFAULT_TEXT_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

_T = TypeVar("_T")

# Transient failures worth retrying. Built defensively so a missing exception
# class in some openai version does not break import.
_RETRYABLE = tuple(
    exc
    for exc in (
        getattr(openai, "RateLimitError", None),
        getattr(openai, "APITimeoutError", None),
        getattr(openai, "APIConnectionError", None),
        getattr(openai, "InternalServerError", None),
    )
    if isinstance(exc, type)
)


def _with_retries(call: Callable[[], _T], *, attempts: int = 4, base_delay: float = 1.5) -> _T:
    """Run ``call``, retrying transient NIM/OpenAI errors with exponential backoff.

    Non-retryable errors (bad request, auth, etc.) propagate immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except _RETRYABLE as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt == attempts - 1:
                break
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 0.5))
    assert last_exc is not None
    raise last_exc


def _strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by Nemotron reasoning models.

    Also handles an unterminated leading block defensively (returns the text
    after the last ``</think>`` if any remain).
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    return cleaned.strip()


class NIMClient:
    """Client for NIM chat, vision, and embedding endpoints."""

    def __init__(self) -> None:
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key.startswith("nvapi-"):
            raise RuntimeError(
                "NVIDIA_API_KEY is missing or malformed (expected an "
                "'nvapi-...' key). Set it in your environment or .env file."
            )
        self._client = OpenAI(
            base_url=os.environ.get("NIM_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key,
        )
        self.vision_model = os.environ.get("NIM_VISION_MODEL", DEFAULT_VISION_MODEL)
        self.text_model = os.environ.get("NIM_TEXT_MODEL", DEFAULT_TEXT_MODEL)
        self.embed_model = os.environ.get("NIM_EMBED_MODEL", DEFAULT_EMBED_MODEL)

    # ------------------------------------------------------------------ text

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Run a text completion on the Nemotron text model.

        ``/no_think`` is prepended to the system prompt to disable the
        model's chain-of-thought output; any ``<think>`` blocks that slip
        through are stripped anyway.
        """
        resp = _with_retries(
            lambda: self._client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": f"/no_think\n{system}"},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        content = resp.choices[0].message.content or ""
        return _strip_reasoning(content)

    # ---------------------------------------------------------------- vision

    def vision(
        self,
        prompt: str,
        image_path: str | Path,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """Send a local image (as a base64 data URL) plus a prompt to the
        NIM vision-language model and return its text response."""
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        resp = _with_retries(
            lambda: self._client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                        ],
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        content = resp.choices[0].message.content or ""
        return _strip_reasoning(content)

    # ------------------------------------------------------------- embedding

    def embed(self, texts: list[str], *, input_type: str = "passage") -> list[list[float]]:
        """Embed a batch of texts with the NIM retrieval embedding model.

        ``input_type`` must be ``"passage"`` for corpus chunks and
        ``"query"`` for search queries (asymmetric embedqa models require
        this; the parameter is passed via ``extra_body``).
        """
        resp = _with_retries(
            lambda: self._client.embeddings.create(
                model=self.embed_model,
                input=texts,
                extra_body={"input_type": input_type, "truncate": "END"},
            )
        )
        # Sort by index defensively; the API documents order preservation
        # but indexes are authoritative.
        data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data]
