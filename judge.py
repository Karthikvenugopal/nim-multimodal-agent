"""Shared LLM-as-judge primitives (NVIDIA NIM text model).

Two roles share this module without a circular import:

* ``parse_judge_json`` + ``JUDGE_SYSTEM`` back the end-of-pipeline benchmark
  judge in :mod:`evaluate` (correctness vs. a gold label **and** faithfulness).
* ``judge_faithfulness`` backs the *in-graph* faithfulness gate in
  :mod:`agent` — a label-free grounding check the agent runs on its own output
  to decide whether to self-correct. An answer that abstains grounds nothing,
  so its gate score is 0 (it must not satisfy the gate for an answerable
  question); abstaining is only the honest final move once corrections are
  exhausted.

This module depends only on a duck-typed ``chat(system, user, ...)`` client,
so it sits below both :mod:`agent` and :mod:`evaluate` in the import graph.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:  # avoid importing nim_client at runtime (keeps the DAG flat)
    from nim_client import NIMClient

JUDGE_SYSTEM = (
    "You are a strict evaluation judge for a retrieval-augmented QA system. "
    "Respond with ONLY a JSON object, no prose, no markdown fences, with "
    "exactly these keys:\n"
    '  "correct": 1 or 0,\n'
    '  "faithfulness": a float from 0.0 to 1.0,\n'
    '  "reason": one short sentence.\n'
    "correct=1 iff the ANSWER conveys the same facts as the GOLD LABEL "
    "(wording may differ). If the gold label says the question is "
    "unanswerable, correct=1 iff the answer abstains. faithfulness is the "
    "fraction of the answer's claims directly supported by the CONTEXT; an "
    "abstention scores 1.0."
)

FAITHFULNESS_SYSTEM = (
    "You are a strict grounding judge. Given a QUESTION, a candidate ANSWER, "
    "and the CONTEXT the answer was supposed to use, judge how well the answer "
    "is supported by the context. Respond with ONLY a JSON object, no prose, "
    "no markdown fences, with exactly these keys:\n"
    '  "answered": 1 or 0,\n'
    '  "faithfulness": a float from 0.0 to 1.0,\n'
    '  "reason": one short sentence.\n'
    "answered=1 if the answer makes a substantive factual claim that addresses "
    "the question; answered=0 if it abstains or says it cannot answer. "
    "faithfulness is the fraction of the answer's claims directly supported by "
    "the context (an abstention is trivially faithful, 1.0)."
)


class JudgeResult(TypedDict):
    correct: int
    faithfulness: float
    reason: str


class FaithResult(TypedDict):
    answered: int
    faithfulness: float
    grounding: float  # faithfulness if answered else 0.0 — the gate score
    reason: str


def _json_candidates(raw: str) -> list[str]:
    """Yield progressively more permissive JSON candidate strings from ``raw``."""
    candidates = [raw.strip()]
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    candidates.append(fenced.strip())
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    return candidates


def _clamp01(value: Any) -> float:
    return min(1.0, max(0.0, float(value)))


def parse_judge_json(raw: str) -> JudgeResult:
    """Parse the benchmark judge's response (correctness + faithfulness).

    Tries plain JSON, then strips markdown fences, then the first ``{...}``
    block. Raises ValueError if no candidate yields valid keys.
    """
    for cand in _json_candidates(raw):
        try:
            obj: dict[str, Any] = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        try:
            correct = 1 if int(obj["correct"]) == 1 else 0
            faith = _clamp01(obj["faithfulness"])
        except (KeyError, TypeError, ValueError):
            continue
        return JudgeResult(
            correct=correct, faithfulness=faith, reason=str(obj.get("reason", ""))
        )
    raise ValueError(f"judge returned unparseable output: {raw[:200]!r}")


def parse_faithfulness(raw: str) -> FaithResult:
    """Parse the in-graph faithfulness judge's response, defensively.

    Unparseable output is treated as ungrounded (grounding 0.0) so the gate
    errs toward attempting a correction rather than accepting an unverified
    answer.
    """
    for cand in _json_candidates(raw):
        try:
            obj: dict[str, Any] = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict) or "faithfulness" not in obj:
            continue
        try:
            faith = _clamp01(obj["faithfulness"])
        except (TypeError, ValueError):
            continue
        answered = 1 if int(obj.get("answered", 1) or 0) == 1 else 0
        return FaithResult(
            answered=answered,
            faithfulness=faith,
            grounding=faith if answered else 0.0,
            reason=str(obj.get("reason", "")),
        )
    return FaithResult(answered=0, faithfulness=0.0, grounding=0.0,
                       reason="unparseable judge output")


def judge_faithfulness(
    client: "NIMClient", question: str, answer: str, context: str
) -> FaithResult:
    """Score how well ``answer`` is grounded in ``context`` (no gold label).

    Returns a :class:`FaithResult` whose ``grounding`` field is the value the
    in-graph gate thresholds on.
    """
    raw = client.chat(
        FAITHFULNESS_SYSTEM,
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nCONTEXT:\n{context}",
        temperature=0.0,
        max_tokens=400,
    )
    return parse_faithfulness(raw)
