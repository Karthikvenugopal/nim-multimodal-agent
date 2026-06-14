"""CLI for the NIM multimodal agentic RAG demo.

Usage::

    python main.py "What is the p95 latency of the VoltEdge Max?"
    python main.py --benchmark
    python main.py --benchmark --no-vision   # text-only baseline
    python main.py --ablation                # vision ON vs OFF comparison
    python main.py --correction              # self-correction loop metrics
    python main.py "..." --correction        # single question, loop trace
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multimodal agentic RAG on NVIDIA NIM (LangGraph)."
    )
    parser.add_argument("question", nargs="?", help="a single question to answer")
    parser.add_argument(
        "--benchmark", action="store_true",
        help="run the full labeled benchmark from corpus/questions.json",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="run the benchmark with and without the vision path and "
             "report the multimodal accuracy lift",
    )
    parser.add_argument(
        "--correction", action="store_true",
        help="enable the faithfulness-gated self-correction loop; with no "
             "question, runs the benchmark and reports loop metrics",
    )
    parser.add_argument(
        "--no-vision", action="store_true",
        help="disable the vision path (text-only baseline)",
    )
    args = parser.parse_args()
    if not args.question and not args.benchmark and not args.ablation and not args.correction:
        parser.error("provide a question, --benchmark, --ablation, or --correction")

    load_dotenv()

    # Imports follow load_dotenv so NIMClient sees .env configuration.
    from agent import answer_question, build_agent, load_corpus, Retriever
    from nim_client import NIMClient

    client = NIMClient()
    print(f"models: vision={client.vision_model}  text={client.text_model}  "
          f"embed={client.embed_model}")
    print("ingesting corpus...")
    chunks = load_corpus()
    retriever = Retriever(client, chunks)
    print(f"ingested {len(chunks)} chunks "
          f"({sum(c.kind == 'text' for c in chunks)} text, "
          f"{sum(c.kind == 'image' for c in chunks)} image)\n")

    if args.ablation:
        from evaluate import run_ablation
        run_ablation(
            client,
            build_agent(client, retriever, enable_vision=True),
            build_agent(client, retriever, enable_vision=False),
        )
        return 0

    if args.correction and not args.question:
        from evaluate import run_correction_benchmark
        agent = build_agent(
            client, retriever, enable_vision=not args.no_vision, enable_correction=True
        )
        run_correction_benchmark(client, agent)
        return 0

    agent = build_agent(
        client, retriever,
        enable_vision=not args.no_vision, enable_correction=args.correction,
    )

    if args.benchmark:
        from evaluate import run_benchmark
        run_benchmark(client, agent)
        return 0

    state = answer_question(agent, args.question)
    print(f"question: {args.question}\n")
    retrieved = ", ".join(f"{c.chunk_id} ({c.kind})" for c in state["chunks"])
    print(f"retrieved: {retrieved}")
    for finding in state.get("image_findings", []):
        print(f"\nvision findings [{finding['image']}]:\n{finding['findings']}")
    if args.correction:
        actions = ", ".join(state.get("actions", [])) or "none"
        history = " -> ".join(f"{f:.2f}" for f in state.get("faith_history", []))
        print(f"\ncorrection loop: attempts={state.get('attempts', 0)}  "
              f"actions=[{actions}]  faithfulness={history or 'n/a'}"
              f"{'  (abstained)' if state.get('abstained') else ''}")
    print(f"\nanswer:\n{state['answer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
