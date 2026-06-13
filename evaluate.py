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

# Question types that can only be answered by reading a figure, so the vision
# path is expected to fire and is what the ablation removes.
VISION_DEPENDENT_TYPES = {"figure", "cross_modal"}

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


def _evaluate(client: NIMClient, agent, *, show_progress: bool = True) -> list[dict[str, Any]]:
    """Run every labeled question through the agent and judge each answer."""
    questions = json.loads(QUESTIONS_PATH.read_text())
    rows: list[dict[str, Any]] = []
    for q in questions:
        start = time.time()
        state = answer_question(agent, q["question"])
        verdict = judge_answer(client, q["question"], q["label"], state["answer"], state)
        row = {
            "id": q["id"],
            "type": q["type"],
            "vision": state.get("used_vision", False),
            "correct": verdict["correct"],
            "faithfulness": verdict["faithfulness"],
            "seconds": time.time() - start,
            "answer": state["answer"],
            "reason": verdict["reason"],
        }
        rows.append(row)
        if show_progress:
            mark = "PASS" if row["correct"] else "FAIL"
            print(f"  [{row['id']}] {mark}  vision={'Y' if row['vision'] else 'n'}  "
                  f"faith={row['faithfulness']:.2f}  ({row['seconds']:.1f}s)")
    return rows


def _by_type(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate accuracy and vision-fire rate per question type."""
    summary: dict[str, dict[str, float]] = {}
    for t in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == t]
        summary[t] = {
            "n": len(group),
            "accuracy": sum(r["correct"] for r in group) / len(group),
            "vision_rate": sum(r["vision"] for r in group) / len(group),
        }
    return summary


def run_benchmark(client: NIMClient, agent) -> dict[str, Any]:
    """Run all labeled questions, print a results table, return aggregates."""
    rows = _evaluate(client, agent)

    header = f"{'id':<5} {'type':<13} {'vision':<7} {'correct':<8} {'faithful':<9} {'sec':>5}"
    sep = "-" * len(header)
    print(f"\n{header}\n{sep}")
    for r in rows:
        print(
            f"{r['id']:<5} {r['type']:<13} {('yes' if r['vision'] else 'no'):<7} "
            f"{r['correct']:<8} {r['faithfulness']:<9.2f} {r['seconds']:>5.1f}"
        )
    print(sep)

    by_type = _by_type(rows)
    print(f"\n{'type':<13} {'n':>3} {'accuracy':>9} {'vision-fire':>12}")
    for t, s in by_type.items():
        print(f"{t:<13} {int(s['n']):>3} {s['accuracy']:>8.1%} {s['vision_rate']:>11.1%}")

    n = len(rows)
    accuracy = sum(r["correct"] for r in rows) / n
    mean_faith = sum(r["faithfulness"] for r in rows) / n
    vis_rows = [r for r in rows if r["type"] in VISION_DEPENDENT_TYPES]
    vis_accuracy = sum(r["correct"] for r in vis_rows) / len(vis_rows) if vis_rows else 0.0
    vis_fire = sum(r["vision"] for r in vis_rows) / len(vis_rows) if vis_rows else 0.0
    print(f"\nquestions:                     {n}")
    print(f"answer accuracy:               {accuracy:.1%}")
    print(f"mean faithfulness:             {mean_faith:.2f}")
    print(f"vision-dependent accuracy:     {vis_accuracy:.1%}  (figure + cross_modal)")
    print(f"vision fired when needed:      {vis_fire:.1%}")
    return {
        "rows": rows,
        "by_type": by_type,
        "accuracy": accuracy,
        "mean_faithfulness": mean_faith,
        "vision_dependent_accuracy": vis_accuracy,
        "vision_fire_rate": vis_fire,
    }


def run_ablation(client: NIMClient, agent_full, agent_blind) -> dict[str, Any]:
    """Run the benchmark with and without the vision path, print the delta.

    Quantifies the multimodal lift: how much answer accuracy depends on the
    agent actually reading the figures versus working from text alone.
    """
    print("=== ablation pass 1/2: vision ENABLED ===")
    full = _evaluate(client, agent_full)
    print("\n=== ablation pass 2/2: vision DISABLED (--no-vision) ===")
    blind = _evaluate(client, agent_blind)

    full_by_type = _by_type(full)
    blind_by_type = _by_type(blind)

    print("\n=== ablation: answer accuracy, vision ON vs OFF ===")
    header = f"{'type':<13} {'n':>3} {'vision-on':>10} {'vision-off':>11} {'delta':>8}"
    sep = "-" * len(header)
    print(f"{header}\n{sep}")
    for t in full_by_type:
        on = full_by_type[t]["accuracy"]
        off = blind_by_type[t]["accuracy"]
        print(f"{t:<13} {int(full_by_type[t]['n']):>3} {on:>9.1%} {off:>10.1%} "
              f"{(on - off) * 100:>+7.1f}")
    print(sep)
    on_all = sum(r["correct"] for r in full) / len(full)
    off_all = sum(r["correct"] for r in blind) / len(blind)
    print(f"{'overall':<13} {len(full):>3} {on_all:>9.1%} {off_all:>10.1%} "
          f"{(on_all - off_all) * 100:>+7.1f}")
    return {
        "full_by_type": full_by_type,
        "blind_by_type": blind_by_type,
        "overall_on": on_all,
        "overall_off": off_all,
    }
