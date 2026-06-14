"""Benchmark harness: runs the labeled question set through the agent and
scores each answer with an LLM-as-judge (Nemotron text model on NIM).

Two metrics per question:

* **correctness** (0/1) — does the answer agree with the gold label?
  For ``unanswerable`` questions, correct means the agent abstained.
* **faithfulness** (0.0–1.0) — is every claim in the answer supported by the
  retrieved context (text passages + vision findings)? An abstention is
  trivially faithful (1.0).

:func:`run_ablation` additionally measures the multimodal lift (vision on vs.
off), and :func:`run_correction_benchmark` measures the in-graph
self-correction loop (trigger rate, recovery rate, faithfulness before/after).
The judge primitives live in :mod:`judge` so the in-graph gate can reuse them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent import AgentState, FAITHFULNESS_THRESHOLD, answer_question
from judge import JUDGE_SYSTEM, JudgeResult, parse_judge_json
from nim_client import NIMClient

QUESTIONS_PATH = Path(__file__).resolve().parent / "corpus" / "questions.json"

# Question types that can only be answered by reading a figure, so the vision
# path is expected to fire and is what the ablation removes.
VISION_DEPENDENT_TYPES = {"figure", "cross_modal"}


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_correction_benchmark(client: NIMClient, agent) -> dict[str, Any]:
    """Run the correction-enabled agent over the set and measure the loop.

    Reports, across the benchmark: correction trigger rate, recovery rate
    (low-faith answers that cleared the gate after correction), and mean
    faithfulness before vs. after correction. The agent must have been built
    with ``enable_correction=True``.
    """
    questions = json.loads(QUESTIONS_PATH.read_text())
    rows: list[dict[str, Any]] = []
    for q in questions:
        start = time.time()
        state = answer_question(agent, q["question"])
        verdict = judge_answer(client, q["question"], q["label"], state["answer"], state)
        history = state.get("faith_history") or [state.get("faithfulness", 0.0)]
        attempts = state.get("attempts", 0)
        abstained = state.get("abstained", False)
        triggered = attempts > 0
        faith_after = history[-1]
        recovered = triggered and not abstained and faith_after >= FAITHFULNESS_THRESHOLD
        result = "abstain" if abstained else ("recover" if triggered else "accept")
        rows.append({
            "id": q["id"], "type": q["type"], "attempts": attempts,
            "actions": state.get("actions", []), "faith_before": history[0],
            "faith_after": faith_after, "triggered": triggered,
            "recovered": recovered, "abstained": abstained, "result": result,
            "correct": verdict["correct"], "seconds": time.time() - start,
        })
        print(f"  [{q['id']}] attempts={attempts} "
              f"[{','.join(state.get('actions', [])) or '-'}]  "
              f"faith {history[0]:.2f}->{faith_after:.2f}  {result:<7} "
              f"correct={verdict['correct']}  ({rows[-1]['seconds']:.1f}s)")

    header = (f"{'id':<5} {'type':<13} {'att':>3} {'actions':<22} "
              f"{'pre':>5} {'post':>5} {'result':<8} {'correct':>7}")
    sep = "-" * len(header)
    print(f"\n{header}\n{sep}")
    for r in rows:
        print(f"{r['id']:<5} {r['type']:<13} {r['attempts']:>3} "
              f"{(','.join(r['actions']) or '-'):<22} {r['faith_before']:>5.2f} "
              f"{r['faith_after']:>5.2f} {r['result']:<8} {r['correct']:>7}")
    print(sep)

    n = len(rows)
    triggered = [r for r in rows if r["triggered"]]
    recovered = [r for r in triggered if r["recovered"]]
    action_counts: dict[str, int] = {}
    for r in rows:
        for a in r["actions"]:
            action_counts[a] = action_counts.get(a, 0) + 1

    recovery_rate = len(recovered) / len(triggered) if triggered else 0.0
    recovery_line = (f"{len(recovered)}/{len(triggered)} = {recovery_rate:.1%}"
                     if triggered else "n/a (nothing triggered)")
    print(f"\nthreshold:                   {FAITHFULNESS_THRESHOLD:.2f}")
    print(f"questions:                   {n}")
    print(f"correction trigger rate:     {len(triggered)}/{n} = {len(triggered) / n:.1%}")
    print(f"recovery rate:               {recovery_line}")
    print(f"mean faithfulness pre-corr:  {_mean([r['faith_before'] for r in triggered]):.2f}  (triggered only)")
    print(f"mean faithfulness post-corr: {_mean([r['faith_after'] for r in triggered]):.2f}  (triggered only)")
    print(f"abstention rate:             {sum(r['abstained'] for r in rows)}/{n} = "
          f"{_mean([float(r['abstained']) for r in rows]):.1%}")
    print(f"final answer accuracy:       {sum(r['correct'] for r in rows)}/{n} = "
          f"{_mean([float(r['correct']) for r in rows]):.1%}")
    print(f"actions fired:               "
          f"{'  '.join(f'{k}={v}' for k, v in sorted(action_counts.items())) or '(none)'}")
    return {
        "rows": rows,
        "trigger_rate": len(triggered) / n,
        "recovery_rate": recovery_rate,
        "mean_faith_before": _mean([r["faith_before"] for r in triggered]),
        "mean_faith_after": _mean([r["faith_after"] for r in triggered]),
        "abstention_rate": _mean([float(r["abstained"]) for r in rows]),
        "accuracy": _mean([float(r["correct"]) for r in rows]),
        "action_counts": action_counts,
    }
