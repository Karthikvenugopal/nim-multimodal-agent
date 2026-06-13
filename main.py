"""CLI for the NIM multimodal agentic RAG demo.

Usage::

    python main.py "What is the p95 latency of the VoltEdge Max?"
    python main.py --benchmark
    python main.py --benchmark --no-vision   # text-only baseline
    python main.py --ablation                # vision ON vs OFF comparison
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
        "--no-vision", action="store_true",
        help="disable the vision path (text-only baseline)",
    )
    args = parser.parse_args()
    if not args.question and not args.benchmark and not args.ablation:
        parser.error("provide a question, --benchmark, or --ablation")

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

    agent = build_agent(client, retriever, enable_vision=not args.no_vision)

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
    print(f"\nanswer:\n{state['answer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
