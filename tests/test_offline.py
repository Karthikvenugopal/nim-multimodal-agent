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
from agent import (
    Chunk,
    _choose_correction,
    _gate_decision,
    _looks_multipart,
    build_agent,
    load_corpus,
)
from evaluate import _by_type, parse_judge_json
from judge import parse_faithfulness
from nim_client import NIMClient, _clean_rewrite, _strip_reasoning, _with_retries


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

    def rewrite_query(self, question: str, **kw) -> str:
        return f"{question} (rewritten)"


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


# -------------------------------------------------- self-correction policy

@pytest.mark.parametrize(
    "grounding, attempts, expected",
    [
        (0.9, 0, "accept"),   # grounded -> done
        (0.3, 0, "correct"),  # low faith, tries remain -> correct
        (0.3, 2, "abstain"),  # low faith, out of tries -> abstain honestly
        (0.7, 1, "accept"),   # exactly at threshold -> accept
    ],
)
def test_gate_decision(grounding: float, attempts: int, expected: str) -> None:
    assert _gate_decision(
        grounding, attempts, threshold=0.7, max_corrections=2
    ) == expected


def test_choose_correction_forces_vision_when_figure_unread() -> None:
    chunks = [_text(), _image()]
    assert _choose_correction(
        chunks, used_vision=False, question="What does the chart show?",
        enable_vision=True,
    ) == "force_vision"


def test_choose_correction_decomposes_multipart_after_vision() -> None:
    chunks = [_text(), _image()]
    # vision already fired -> a multi-part question should decompose
    action = _choose_correction(
        chunks, used_vision=True,
        question="Which variant is fastest, and how many streams does it support?",
        enable_vision=True,
    )
    assert action == "decompose"


def test_choose_correction_defaults_to_rewrite() -> None:
    chunks = [_text(0), _text(1)]  # no image, single-part question
    assert _choose_correction(
        chunks, used_vision=False, question="What port is used?", enable_vision=True,
    ) == "rewrite_query"


def test_looks_multipart() -> None:
    assert _looks_multipart("Which variant is fastest, and how many streams?") is True
    assert _looks_multipart("What port is the management API on?") is False


def test_correction_loop_recovers_then_abstains(monkeypatch) -> None:
    """End-to-end loop on stubs: a low-faith answer triggers a corrective
    action; once attempts are exhausted the agent abstains."""
    from agent import ABSTAIN_ANSWER

    # Faithfulness judge always returns low grounding -> never clears the gate.
    monkeypatch.setattr(
        "agent.judge_faithfulness",
        lambda client, q, a, ctx: {"answered": 1, "faithfulness": 0.1,
                                    "grounding": 0.1, "reason": "stub"},
    )
    client = StubClient()
    agent = build_agent(
        client, StubRetriever([(_image(), 0.6), (_text(), 0.4)]),
        enable_correction=True,
    )
    state = agent.invoke({"question": "What does the chart show?"},
                         {"recursion_limit": 50})
    assert state["attempts"] == 2  # MAX_CORRECTIONS default
    assert state["abstained"] is True
    assert state["answer"] == ABSTAIN_ANSWER
    assert state["actions"][0] == "force_vision"  # figure present, vision unread


def test_parse_faithfulness_defensive() -> None:
    grounded = parse_faithfulness('{"answered": 1, "faithfulness": 0.9, "reason": "ok"}')
    assert grounded["grounding"] == pytest.approx(0.9)
    # an abstention grounds nothing even though faithfulness is high
    abstained = parse_faithfulness('{"answered": 0, "faithfulness": 1.0, "reason": "abstain"}')
    assert abstained["grounding"] == 0.0
    # unparseable -> treated as ungrounded so the gate errs toward correcting
    assert parse_faithfulness("garbage")["grounding"] == 0.0


def test_clean_rewrite_falls_back() -> None:
    assert _clean_rewrite("", "original q") == "original q"
    assert _clean_rewrite("  rewritten query \nextra", "orig") == "rewritten query"
