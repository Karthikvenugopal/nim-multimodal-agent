"""Render the compiled LangGraph agent to docs/graph.png and docs/graph.mmd.

The graph structure is independent of the live NIM models, so lightweight
stubs stand in for the client and retriever. PNG rendering uses LangGraph's
Mermaid renderer (remote mermaid.ink); if that is unavailable the Mermaid
source is still written and embedded in the README.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import build_agent  # noqa: E402

DOCS = ROOT / "docs"


class _Stub:
    """Stand-in for NIMClient / Retriever; graph rendering invokes no nodes."""

    def search(self, *args, **kwargs):
        return []

    def embed(self, *args, **kwargs):
        return [[0.0]]

    def vision(self, *args, **kwargs):
        return ""

    def chat(self, *args, **kwargs):
        return ""


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    # Render the full graph including the self-correction loop.
    graph = build_agent(_Stub(), _Stub(), enable_correction=True).get_graph()

    mermaid = graph.draw_mermaid()
    (DOCS / "graph.mmd").write_text(mermaid)
    print("wrote docs/graph.mmd\n")
    print(mermaid)

    try:
        (DOCS / "graph.png").write_bytes(graph.draw_mermaid_png())
        print("wrote docs/graph.png")
    except Exception as exc:  # network/render backend unavailable
        print(f"PNG render skipped ({exc!s}); README embeds the Mermaid source instead")


if __name__ == "__main__":
    main()
