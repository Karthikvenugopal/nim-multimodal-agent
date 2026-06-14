"""Multimodal agentic RAG pipeline as a LangGraph state graph.

Base flow (``enable_correction=False`` — unchanged from the original)::

    retrieve ──(relevant image queued?)──> analyze_images ──> generate ──> END
        └──────────(text only)──────────────────────────────────^

* ``retrieve`` embeds the query and pulls the top-k corpus chunks
  (text passages and image captions) by cosine similarity. An image chunk is
  queued for vision analysis only when it is *relevant enough*: ranked first
  overall, or above an absolute similarity threshold.
* ``analyze_images`` sends each queued image (base64) to the NIM
  vision-language model to extract the facts it contains.
* ``generate`` produces a grounded answer with the Nemotron text model from
  the text passages plus the vision findings, abstaining when the context is
  insufficient.

``enable_vision=False`` removes the vision route entirely (the "blind" baseline
used by the ablation in :mod:`evaluate`).

Self-correction loop (``enable_correction=True`` — additive)::

    ... generate ──> judge_gate ──(grounded?)──────────────> END
                         │  └──(low faith, tries left)──> correct ──┐
                         │                                          │
                         └──(low faith, out of tries)──> abstain    │
                                  ^                                  │
                                  └──────── generate <──────────────┘

The end-of-pipeline faithfulness judge becomes an *in-graph control signal*
(Corrective-RAG / Self-RAG / Reflexion): after generating, the agent judges
its own answer's grounding and, if it falls below ``FAITHFULNESS_THRESHOLD``,
takes one corrective action and regenerates. To make that loop the thing that
grounds figure answers (and therefore measurable), correction mode is
*text-first*: the first pass skips vision, and the gate escalates to the vision
tool, a query rewrite, or question decomposition only when its own check fails.
After ``MAX_CORRECTIONS`` attempts it abstains honestly rather than return a
low-faith answer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from judge import judge_faithfulness
from nim_client import NIMClient

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
TOP_K = 4
# An image chunk is sent to the vision model only if it is the top-ranked
# retrieval or its cosine similarity to the query is at least this value.
IMAGE_SCORE_THRESHOLD = 0.5

# Self-correction loop configuration (overridable via environment).
FAITHFULNESS_THRESHOLD = float(os.environ.get("FAITHFULNESS_THRESHOLD", "0.7"))
MAX_CORRECTIONS = int(os.environ.get("MAX_CORRECTIONS", "2"))

GENERATION_SYSTEM = (
    "You are a careful assistant answering questions about internal company "
    "documents. Answer ONLY from the provided context (text passages and "
    "image analysis findings). Be concise and cite the figure or document "
    "you used. If the context does not contain the answer, reply exactly: "
    "\"I cannot answer this from the provided documents.\" Never guess or "
    "use outside knowledge."
)

ABSTAIN_ANSWER = (
    "I cannot answer this from the provided documents with sufficient grounding."
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
    # --- self-correction loop telemetry (only populated when enabled) ---
    attempts: int  # number of corrective actions taken
    actions: list[str]  # which corrective actions fired, in order
    faith_history: list[float]  # gate grounding score after each generation
    faithfulness: float  # latest gate grounding score
    gate_reason: str  # judge's one-line rationale for the latest score
    decomposition: list[dict[str, str]]  # sub-question/answer pairs (decompose)
    rewritten_query: str  # last reformulated query (rewrite_query)
    abstained: bool  # True if the loop gave up and abstained


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


# ----------------------------------------------------------------- helpers

def _queue_images(hits: list[tuple[Chunk, float]]) -> list[Chunk]:
    """Apply the relevance gate to retrieval hits, returning images to analyze."""
    return [
        chunk
        for rank, (chunk, score) in enumerate(hits)
        if chunk.kind == "image" and (rank == 0 or score >= IMAGE_SCORE_THRESHOLD)
    ]


def _vision_prompt(question: str) -> str:
    """Prompt the VLM to transcribe a figure's facts for a given question."""
    return (
        "You are analyzing a figure from internal documentation to help answer "
        f"this question:\n  {question}\n\n"
        "Describe exactly what the figure shows, including every label, numeric "
        "value, and relationship visible. Then state which facts (if any) are "
        "relevant to the question. Report only what is actually visible in the "
        "image."
    )


def _analyze_image(client: NIMClient, question: str, chunk: Chunk) -> dict[str, str]:
    """Run the vision model on one image chunk and return a findings record."""
    return {
        "image": chunk.source,
        "findings": client.vision(_vision_prompt(question), CORPUS_DIR / chunk.source),
    }


def _context_parts(state: AgentState) -> list[str]:
    """Assemble the labeled context blocks the generator and judge both see."""
    parts: list[str] = []
    for chunk in state.get("chunks", []):
        if chunk.kind == "text":
            parts.append(f"[text passage from {chunk.source}]\n{chunk.text}")
    for item in state.get("image_findings", []):
        parts.append(
            f"[vision-model analysis of figure {item['image']}]\n{item['findings']}"
        )
    for sub in state.get("decomposition", []):
        parts.append(f"[sub-question] {sub['q']}\n[sub-answer] {sub['a']}")
    return parts


def _looks_multipart(question: str) -> bool:
    """Label-free heuristic: does the question ask for two things at once?"""
    lowered = question.lower()
    return " and " in lowered and len(question.split()) >= 8


def _gate_decision(
    grounding: float, attempts: int, *, threshold: float, max_corrections: int
) -> Literal["accept", "correct", "abstain"]:
    """Faithfulness gate: accept a grounded answer, correct, or give up."""
    if grounding >= threshold:
        return "accept"
    if attempts >= max_corrections:
        return "abstain"
    return "correct"


def _choose_correction(
    chunks: list[Chunk], used_vision: bool, question: str, *, enable_vision: bool
) -> Literal["force_vision", "decompose", "rewrite_query"]:
    """Explainable policy mapping a failed answer to one corrective action."""
    has_image = any(c.kind == "image" for c in chunks)
    if enable_vision and has_image and not used_vision:
        return "force_vision"
    if _looks_multipart(question):
        return "decompose"
    return "rewrite_query"


def _decompose(client: NIMClient, question: str, context: str) -> list[dict[str, str]]:
    """Split a multi-part question into sub-questions and answer each from context."""
    raw = client.chat(
        "Break the question into 2 or 3 minimal sub-questions, each answerable "
        "on its own. Output one sub-question per line, no numbering or extra text.",
        question,
        temperature=0.2,
        max_tokens=200,
    )
    sub_questions = [re.sub(r"^[\s\-*\d.)]+", "", line).strip() for line in raw.splitlines()]
    sub_questions = [s for s in sub_questions if len(s.split()) >= 3][:3]
    pairs: list[dict[str, str]] = []
    for sub_q in sub_questions:
        answer = client.chat(
            "Answer the question using ONLY the context. If the context does not "
            "contain it, reply 'not in context'. Answer in one sentence.",
            f"Context:\n{context}\n\nQuestion: {sub_q}",
            temperature=0.1,
            max_tokens=200,
        )
        pairs.append({"q": sub_q, "a": answer})
    return pairs


def build_agent(
    client: NIMClient,
    retriever: Retriever,
    *,
    enable_vision: bool = True,
    enable_correction: bool = False,
):
    """Compile and return the LangGraph agent.

    * ``enable_vision=False`` removes the vision route (ablation baseline).
    * ``enable_correction=True`` adds the faithfulness-gated self-correction
      loop and runs the first pass text-first so the loop is the mechanism that
      grounds figure/cross-modal answers.
    """

    def retrieve(state: AgentState) -> AgentState:
        """Retrieve top-k chunks and queue images that pass the relevance gate."""
        hits = retriever.search(state["question"])
        return {"chunks": [c for c, _ in hits], "image_queue": _queue_images(hits)}

    def route_after_retrieve(state: AgentState) -> str:
        # Correction mode is text-first: defer vision to the corrective action.
        if enable_vision and not enable_correction and state.get("image_queue"):
            return "analyze_images"
        return "generate"

    def analyze_images(state: AgentState) -> AgentState:
        """Send each queued image to the NIM vision model."""
        findings = [_analyze_image(client, state["question"], c) for c in state["image_queue"]]
        return {"image_findings": findings, "used_vision": bool(findings)}

    def generate(state: AgentState) -> AgentState:
        """Compose the grounded answer from text passages + vision findings."""
        parts = _context_parts(state)
        context = "\n\n---\n\n".join(parts) if parts else "(no context retrieved)"
        answer = client.chat(
            GENERATION_SYSTEM,
            f"Context:\n\n{context}\n\nQuestion: {state['question']}",
        )
        return {"answer": answer, "used_vision": state.get("used_vision", False)}

    def judge_gate(state: AgentState) -> AgentState:
        """Score the answer's grounding; this is the in-graph control signal."""
        context = "\n\n---\n\n".join(_context_parts(state)) or "(no context)"
        verdict = judge_faithfulness(client, state["question"], state["answer"], context)
        grounding = verdict["grounding"]
        # Backstop: a canonical abstention grounds no claim, whatever the judge said.
        if "cannot answer" in state["answer"].lower():
            grounding = 0.0
        return {
            "faithfulness": grounding,
            "faith_history": state.get("faith_history", []) + [grounding],
            "gate_reason": verdict["reason"],
        }

    def route_after_gate(state: AgentState) -> str:
        return _gate_decision(
            state["faithfulness"],
            state.get("attempts", 0),
            threshold=FAITHFULNESS_THRESHOLD,
            max_corrections=MAX_CORRECTIONS,
        )

    def correct(state: AgentState) -> AgentState:
        """Take one corrective action based on the explainable policy."""
        action = _choose_correction(
            state["chunks"],
            state.get("used_vision", False),
            state["question"],
            enable_vision=enable_vision,
        )
        update: AgentState = {
            "attempts": state.get("attempts", 0) + 1,
            "actions": state.get("actions", []) + [action],
        }
        if action == "force_vision":
            top_image = next((c for c in state["chunks"] if c.kind == "image"), None)
            findings = list(state.get("image_findings", []))
            if top_image is not None:
                findings.append(_analyze_image(client, state["question"], top_image))
            update["image_findings"] = findings
            update["used_vision"] = True
        elif action == "decompose":
            context = "\n\n---\n\n".join(_context_parts(state)) or "(no context)"
            update["decomposition"] = state.get("decomposition", []) + _decompose(
                client, state["question"], context
            )
        else:  # rewrite_query
            new_query = client.rewrite_query(state["question"])
            hits = retriever.search(new_query)
            update["chunks"] = [c for c, _ in hits]
            update["image_queue"] = _queue_images(hits)
            update["rewritten_query"] = new_query
        return update

    def give_up(state: AgentState) -> AgentState:
        """Out of corrections and still ungrounded — abstain honestly."""
        return {"answer": ABSTAIN_ANSWER, "abstained": True}

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

    if enable_correction:
        graph.add_node("judge_gate", judge_gate)
        graph.add_node("correct", correct)
        graph.add_node("abstain", give_up)
        graph.add_edge("generate", "judge_gate")
        graph.add_conditional_edges(
            "judge_gate",
            route_after_gate,
            {"accept": END, "correct": "correct", "abstain": "abstain"},
        )
        graph.add_edge("correct", "generate")
        graph.add_edge("abstain", END)
    else:
        graph.add_edge("generate", END)

    return graph.compile()


def answer_question(agent, question: str) -> AgentState:
    """Run the compiled agent on a single question and return final state."""
    # Generous recursion limit so the correction loop never trips it.
    result: AgentState = agent.invoke({"question": question}, {"recursion_limit": 50})
    return result
