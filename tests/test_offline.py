"""Offline unit tests — no NIM API key or network required.

These exercise corpus loading, defensive judge-JSON parsing, graph routing
(including the relevance gate and the --no-vision ablation), the retry helper,
and the response cleaning. The NIM models are replaced with stubs so the whole
suite runs in CI without credentials.
"""

from __future__ import annotations

import httpx
import openai
import pytest

import nim_client
from agent import Chunk, build_agent, load_corpus
from evaluate import _by_type, parse_judge_json
from nim_client import NIMClient, _strip_reasoning, _with_retries


# --------------------------------------------------------------------- stubs

class StubClient:
    """Records vision/chat calls; returns canned responses."""

    def __init__(self) -> None:
        self.vision_calls = 0

    def vision(self, prompt: str, path) -> str:
        self.vision_calls += 1
        return f"stub findings for {path}"

    def chat(self, system: str, user: str, **kw) -> str:
        return "stub answer"


class StubRetriever:
    """Returns a fixed list of (chunk, score) hits."""

    def __init__(self, hits: list[tuple[Chunk, float]]) -> None:
        self._hits = hits

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        return self._hits


def _text(i: int = 0) -> Chunk:
    return Chunk(chunk_id=f"t{i}", kind="text", text=f"passage {i}", source="doc.md")


def _image(i: int = 0) -> Chunk:
    return Chunk(chunk_id=f"img{i}", kind="image", text=f"caption {i}",
                 source=f"images/fig_{i}.png")


# ------------------------------------------------------------------- corpus

def test_load_corpus_has_text_and_images() -> None:
    chunks = load_corpus()
    kinds = {c.kind for c in chunks}
    assert kinds == {"text", "image"}
    assert sum(c.kind == "image" for c in chunks) == 5
    assert sum(c.kind == "text" for c in chunks) >= 8


# -------------------------------------------------------------- judge parser

@pytest.mark.parametrize(
    "raw, correct, faith",
    [
        ('{"correct": 1, "faithfulness": 0.9, "reason": "x"}', 1, 0.9),
        ('```json\n{"correct": 0, "faithfulness": 1.4, "reason": "y"}\n```', 0, 1.0),
        ('blah {"correct": "1", "faithfulness": "0.5"} trailing', 1, 0.5),
        ('{"correct": 0, "faithfulness": -3, "reason": "z"}', 0, 0.0),
    ],
)
def test_parse_judge_json_variants(raw: str, correct: int, faith: float) -> None:
    result = parse_judge_json(raw)
    assert result["correct"] == correct
    assert result["faithfulness"] == pytest.approx(faith)


def test_parse_judge_json_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_judge_json("not json at all")


# ------------------------------------------------------------------ routing

def test_vision_fires_when_image_ranks_first() -> None:
    client = StubClient()
    agent = build_agent(client, StubRetriever([(_image(), 0.6), (_text(), 0.4)]))
    state = agent.invoke({"question": "q"})
    assert state["used_vision"] is True
    assert client.vision_calls == 1
    assert state["answer"] == "stub answer"


def test_vision_skipped_for_low_scoring_non_top_image() -> None:
    client = StubClient()
    agent = build_agent(client, StubRetriever([(_text(), 0.5), (_image(), 0.4)]))
    state = agent.invoke({"question": "q"})
    assert state.get("used_vision", False) is False
    assert client.vision_calls == 0


def test_vision_fires_for_high_scoring_non_top_image() -> None:
    client = StubClient()
    agent = build_agent(client, StubRetriever([(_text(), 0.6), (_image(), 0.55)]))
    state = agent.invoke({"question": "q"})
    assert state["used_vision"] is True
    assert client.vision_calls == 1


def test_no_vision_mode_never_calls_vision() -> None:
    client = StubClient()
    agent = build_agent(
        client, StubRetriever([(_image(), 0.9), (_text(), 0.4)]), enable_vision=False
    )
    state = agent.invoke({"question": "q"})
    assert state.get("used_vision", False) is False
    assert client.vision_calls == 0


# ----------------------------------------------------------- type aggregation

def test_by_type_aggregation() -> None:
    rows = [
        {"type": "text", "correct": 1, "vision": False},
        {"type": "text", "correct": 0, "vision": False},
        {"type": "figure", "correct": 1, "vision": True},
    ]
    summary = _by_type(rows)
    assert summary["text"]["n"] == 2
    assert summary["text"]["accuracy"] == pytest.approx(0.5)
    assert summary["figure"]["vision_rate"] == pytest.approx(1.0)


# ----------------------------------------------------------------- retries

def test_with_retries_retries_transient_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(nim_client.time, "sleep", lambda _s: None)
    transient = openai.APITimeoutError(httpx.Request("POST", "https://x"))
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise transient
        return "ok"

    assert _with_retries(flaky, attempts=3, base_delay=0.0) == "ok"
    assert calls["n"] == 2


def test_with_retries_propagates_non_retryable() -> None:
    def boom() -> str:
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        _with_retries(boom, attempts=3, base_delay=0.0)


# -------------------------------------------------------------- misc helpers

def test_strip_reasoning() -> None:
    assert _strip_reasoning("<think>hmm</think>final") == "final"
    assert _strip_reasoning("no tags here") == "no tags here"
    assert _strip_reasoning("a<think>x</think>b<think>y</think>c") == "abc"


def test_missing_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        NIMClient()
