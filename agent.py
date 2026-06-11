"""Multimodal agentic RAG pipeline as a LangGraph state graph.

Flow::

    retrieve ──(any image chunks?)──> analyze_images ──> generate
        └──────────(text only)─────────────────────────────^

* ``retrieve`` embeds the query and pulls the top-k corpus chunks
  (text passages and image captions) by cosine similarity. An image chunk is
  queued for vision analysis only when it is *relevant enough*: ranked first
  overall, or above an absolute similarity threshold. This keeps the vision
  route selective instead of firing on every query.
* ``analyze_images`` fires only when relevant image chunks were retrieved:
  it sends the actual images (base64) to the NIM vision model to extract the
  facts relevant to the question.
* ``generate`` produces a grounded answer with the Nemotron text model from
  the text passages plus the vision findings, abstaining when the context
  is insufficient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from nim_client import NIMClient

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
TOP_K = 4
# An image chunk is sent to the vision model only if it is the top-ranked
# retrieval or its cosine similarity to the query is at least this value.
IMAGE_SCORE_THRESHOLD = 0.5

GENERATION_SYSTEM = (
    "You are a careful assistant answering questions about internal company "
    "documents. Answer ONLY from the provided context (text passages and "
    "image analysis findings). Be concise and cite the figure or document "
    "you used. If the context does not contain the answer, reply exactly: "
    "\"I cannot answer this from the provided documents.\" Never guess or "
    "use outside knowledge."
)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit: a text passage or an image caption."""

    chunk_id: str
    kind: Literal["text", "image"]
    text: str  # passage text, or the image caption
    source: str  # doc filename or image path relative to corpus/


class AgentState(TypedDict, total=False):
    """State carried through the LangGraph graph."""

    question: str
    chunks: list[Chunk]
    image_queue: list[Chunk]  # retrieved images relevant enough to analyze
    image_findings: list[dict[str, str]]  # {"image": path, "findings": text}
    answer: str
    used_vision: bool


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    """Load text docs (split into paragraph chunks) and image captions."""
    chunks: list[Chunk] = []
    for doc in sorted((corpus_dir / "docs").glob("*.md")):
        paragraphs = [p.strip() for p in doc.read_text().split("\n\n")]
        for i, para in enumerate(p for p in paragraphs if len(p) >= 40):
            chunks.append(
                Chunk(chunk_id=f"{doc.stem}#{i}", kind="text", text=para, source=doc.name)
            )
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    for entry in manifest.get("images", []):
        image_path = corpus_dir / entry["file"]
        if not image_path.exists():
            raise FileNotFoundError(f"manifest references missing image: {image_path}")
        chunks.append(
            Chunk(
                chunk_id=Path(entry["file"]).stem,
                kind="image",
                text=entry["caption"],
                source=entry["file"],
            )
        )
    return chunks


class Retriever:
    """In-memory dense retriever over corpus chunks (NIM embeddings + cosine)."""

    def __init__(self, client: NIMClient, chunks: list[Chunk]) -> None:
        self._client = client
        self._chunks = chunks
        vectors = client.embed([c.text for c in chunks], input_type="passage")
        matrix = np.asarray(vectors, dtype=np.float32)
        self._matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        """Return the top-k (chunk, cosine score) pairs for the query."""
        qvec = np.asarray(
            self._client.embed([query], input_type="query")[0], dtype=np.float32
        )
        qvec /= np.linalg.norm(qvec)
        scores = self._matrix @ qvec
        top = np.argsort(scores)[::-1][:k]
        return [(self._chunks[i], float(scores[i])) for i in top]


def build_agent(client: NIMClient, retriever: Retriever):
    """Compile and return the LangGraph agent."""

    def retrieve(state: AgentState) -> AgentState:
        """Retrieve top-k chunks and queue images that pass the relevance gate."""
        hits = retriever.search(state["question"])
        image_queue = [
            chunk
            for rank, (chunk, score) in enumerate(hits)
            if chunk.kind == "image" and (rank == 0 or score >= IMAGE_SCORE_THRESHOLD)
        ]
        return {"chunks": [c for c, _ in hits], "image_queue": image_queue}

    def route_after_retrieve(state: AgentState) -> str:
        return "analyze_images" if state.get("image_queue") else "generate"

    def analyze_images(state: AgentState) -> AgentState:
        """Send each queued image to the NIM vision model."""
        findings: list[dict[str, str]] = []
        for chunk in state["image_queue"]:
            prompt = (
                "You are analyzing a figure from internal documentation to "
                "help answer this question:\n"
                f"  {state['question']}\n\n"
                "Describe exactly what the figure shows, including every "
                "label, numeric value, and relationship visible. Then state "
                "which facts (if any) are relevant to the question. Report "
                "only what is actually visible in the image."
            )
            findings.append(
                {
                    "image": chunk.source,
                    "findings": client.vision(prompt, CORPUS_DIR / chunk.source),
                }
            )
        return {"image_findings": findings, "used_vision": bool(findings)}

    def generate(state: AgentState) -> AgentState:
        """Compose the grounded answer from text passages + vision findings."""
        parts: list[str] = []
        for chunk in state["chunks"]:
            if chunk.kind == "text":
                parts.append(f"[text passage from {chunk.source}]\n{chunk.text}")
        for item in state.get("image_findings", []):
            parts.append(
                f"[vision-model analysis of figure {item['image']}]\n{item['findings']}"
            )
        context = "\n\n---\n\n".join(parts) if parts else "(no context retrieved)"
        answer = client.chat(
            GENERATION_SYSTEM,
            f"Context:\n\n{context}\n\nQuestion: {state['question']}",
        )
        return {"answer": answer, "used_vision": state.get("used_vision", False)}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("analyze_images", analyze_images)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"analyze_images": "analyze_images", "generate": "generate"},
    )
    graph.add_edge("analyze_images", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def answer_question(agent, question: str) -> AgentState:
    """Run the compiled agent on a single question and return final state."""
    result: AgentState = agent.invoke({"question": question})
    return result
