"""Benchmark harness: runs the labeled question set through the agent and
scores each answer with an LLM-as-judge (Nemotron text model on NIM).

Two metrics per question:

* **correctness** (0/1) — does the answer agree with the gold label?
  For ``unanswerable`` questions, correct means the agent abstained.
* **faithfulness** (0.0–1.0) — is every claim in the answer supported by the
  retrieved context (text passages + vision findings)? An abstention is
  trivially faithful (1.0).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, TypedDict

from agent import AgentState, answer_question
from nim_client import NIMClient

QUESTIONS_PATH = Path(__file__).resolve().parent / "corpus" / "questions.json"

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


class JudgeResult(TypedDict):
    correct: int
    faithfulness: float
    reason: str


def parse_judge_json(raw: str) -> JudgeResult:
    """Defensively parse the judge's response into a JudgeResult.

    Tries plain JSON, then strips markdown fences, then falls back to the
    first ``{...}`` block in the text. Raises ValueError if nothing parses.
    """
    candidates = [raw.strip()]
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    candidates.append(fenced.strip())
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        try:
            obj: dict[str, Any] = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        try:
            correct = 1 if int(obj["correct"]) == 1 else 0
            faith = min(1.0, max(0.0, float(obj["faithfulness"])))
        except (KeyError, TypeError, ValueError):
            continue
        return JudgeResult(
            correct=correct, faithfulness=faith, reason=str(obj.get("reason", ""))
        )
    raise ValueError(f"judge returned unparseable output: {raw[:200]!r}")


def judge_answer(
    client: NIMClient, question: str, label: str, answer: str, state: AgentState
) -> JudgeResult:
    """Score one answer for correctness (vs. label) and faithfulness (vs. context)."""
    context_parts = [
        f"[{c.kind} chunk from {c.source}]\n{c.text}" for c in state.get("chunks", [])
    ]
    context_parts += [
        f"[vision findings for {f['image']}]\n{f['findings']}"
        for f in state.get("image_findings", [])
    ]
    user = (
        f"QUESTION:\n{question}\n\n"
        f"GOLD LABEL:\n{label}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"CONTEXT:\n{chr(10).join(context_parts)}"
    )
    raw = client.chat(JUDGE_SYSTEM, user, temperature=0.0, max_tokens=512)
    return parse_judge_json(raw)


def run_benchmark(client: NIMClient, agent) -> dict[str, Any]:
    """Run all labeled questions, print a results table, return aggregates."""
    questions = json.loads(QUESTIONS_PATH.read_text())
    rows: list[dict[str, Any]] = []

    for q in questions:
        start = time.time()
        state = answer_question(agent, q["question"])
        verdict = judge_answer(client, q["question"], q["label"], state["answer"], state)
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "vision": state.get("used_vision", False),
                "correct": verdict["correct"],
                "faithfulness": verdict["faithfulness"],
                "seconds": time.time() - start,
                "answer": state["answer"],
                "reason": verdict["reason"],
            }
        )
        mark = "PASS" if verdict["correct"] else "FAIL"
        print(f"  [{q['id']}] {mark}  vision={'Y' if rows[-1]['vision'] else 'n'}  "
              f"faith={verdict['faithfulness']:.2f}  ({rows[-1]['seconds']:.1f}s)")

    # ----------------------------------------------------------- summary
    header = f"{'id':<4} {'type':<13} {'vision':<7} {'correct':<8} {'faithful':<9} {'sec':>5}"
    sep = "-" * len(header)
    print(f"\n{header}\n{sep}")
    for r in rows:
        print(
            f"{r['id']:<4} {r['type']:<13} {('yes' if r['vision'] else 'no'):<7} "
            f"{r['correct']:<8} {r['faithfulness']:<9.2f} {r['seconds']:>5.1f}"
        )
    print(sep)

    n = len(rows)
    accuracy = sum(r["correct"] for r in rows) / n
    mean_faith = sum(r["faithfulness"] for r in rows) / n
    figure_rows = [r for r in rows if r["type"] == "figure"]
    vision_fire_rate = (
        sum(r["vision"] for r in figure_rows) / len(figure_rows) if figure_rows else 0.0
    )
    figure_accuracy = (
        sum(r["correct"] for r in figure_rows) / len(figure_rows) if figure_rows else 0.0
    )
    print(f"questions:                  {n}")
    print(f"answer accuracy:            {accuracy:.1%}")
    print(f"mean faithfulness:          {mean_faith:.2f}")
    print(f"figure-question accuracy:   {figure_accuracy:.1%}")
    print(f"vision fired on figure Qs:  {vision_fire_rate:.1%}")
    return {
        "rows": rows,
        "accuracy": accuracy,
        "mean_faithfulness": mean_faith,
        "figure_accuracy": figure_accuracy,
        "vision_fire_rate": vision_fire_rate,
    }
