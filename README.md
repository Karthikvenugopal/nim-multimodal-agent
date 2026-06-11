# nim-multimodal-agent

A **multimodal agentic RAG pipeline** built on [NVIDIA NIM](https://build.nvidia.com)
with a benchmark-style evaluation layer. A LangGraph agent retrieves over a mixed
text + image corpus, routes retrieved figures through a NIM vision-language model
to extract the facts they contain, and generates grounded answers with a Nemotron
text model — scored by an LLM-as-judge benchmark for correctness and faithfulness.

```
retrieve ──(image chunk retrieved?)──> analyze_images ──> generate
    └──────────(text only)────────────────────────────────────^
```

- **Genuinely multimodal**: the figure data in `corpus/images/` (latency
  benchmarks, revenue mix, GPU utilization, pipeline diagram, error rates)
  exists *only in the pixels* — figure questions cannot be answered from the
  text, so they exercise the real vision path (base64 image → NIM VLM).
- **Genuinely agentic**: a compiled LangGraph `StateGraph` with a conditional
  edge that routes to vision analysis only when a retrieved image passes a
  relevance gate (top-ranked, or above an absolute similarity threshold) —
  in the benchmark below, vision fires on 5/5 figure questions and 0/6
  text/unanswerable questions.
- **Honest evaluation**: an 11-question labeled set (text-answerable,
  figure-only, and one unanswerable question that tests abstention), scored
  for answer correctness and context-faithfulness by a Nemotron judge.

## Models (NVIDIA NIM, OpenAI-compatible API)

| Role | Default model | Env var |
|---|---|---|
| Vision-language | `nvidia/nemotron-nano-12b-v2-vl` | `NIM_VISION_MODEL` |
| Generation + judge | `nvidia/llama-3.3-nemotron-super-49b-v1.5` | `NIM_TEXT_MODEL` |
| Retrieval embeddings | `nvidia/llama-nemotron-embed-1b-v2` | `NIM_EMBED_MODEL` |

All calls go through `https://integrate.api.nvidia.com/v1` via the `openai`
SDK. The NIM catalog changes often — override any model with the env vars
above (see `.env.example`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # paste your nvapi- key from https://build.nvidia.com
```

## Usage

```bash
# single question (multimodal: retrieves the figure, fires the vision model)
python main.py "What is the p95 inference latency of the VoltEdge Max on ResNet-50?"

# full labeled benchmark with LLM-as-judge scoring
python main.py --benchmark
```

## Repo layout

```
corpus/docs/         3 markdown docs (text facts)
corpus/images/       5 chart/diagram PNGs (figure-only facts)
corpus/manifest.json image captions used for retrieval (values withheld)
corpus/questions.json 11 labeled benchmark questions
nim_client.py        NIM chat / vision / embedding client
agent.py             LangGraph graph: retrieve → route → vision → generate
evaluate.py          benchmark harness + LLM-as-judge
main.py              CLI
scripts/make_images.py  regenerates the corpus figures (matplotlib)
```

## Benchmark results

Real output of `python main.py --benchmark` against the live NIM API:

```
models: vision=nvidia/nemotron-nano-12b-v2-vl  text=nvidia/llama-3.3-nemotron-super-49b-v1.5  embed=nvidia/llama-nemotron-embed-1b-v2
ingesting corpus...
ingested 16 chunks (11 text, 5 image)

  [T1] PASS  vision=n  faith=1.00  (3.0s)
  [T2] PASS  vision=n  faith=1.00  (2.9s)
  [T3] PASS  vision=n  faith=1.00  (5.2s)
  [T4] PASS  vision=n  faith=1.00  (4.3s)
  [T5] PASS  vision=n  faith=1.00  (35.7s)
  [F1] PASS  vision=Y  faith=1.00  (17.0s)
  [F2] PASS  vision=Y  faith=1.00  (21.1s)
  [F3] PASS  vision=Y  faith=1.00  (12.2s)
  [F4] PASS  vision=Y  faith=1.00  (20.6s)
  [F5] PASS  vision=Y  faith=1.00  (14.2s)
  [U1] PASS  vision=n  faith=1.00  (33.2s)

id   type          vision  correct  faithful    sec
---------------------------------------------------
T1   text          no      1        1.00        3.0
T2   text          no      1        1.00        2.9
T3   text          no      1        1.00        5.2
T4   text          no      1        1.00        4.3
T5   text          no      1        1.00       35.7
F1   figure        yes     1        1.00       17.0
F2   figure        yes     1        1.00       21.1
F3   figure        yes     1        1.00       12.2
F4   figure        yes     1        1.00       20.6
F5   figure        yes     1        1.00       14.2
U1   unanswerable  no      1        1.00       33.2
---------------------------------------------------
questions:                  11
answer accuracy:            100.0%
mean faithfulness:          1.00
figure-question accuracy:   100.0%
vision fired on figure Qs:  100.0%
```

Metric notes: **correct** is judged against the gold label (for the
unanswerable question U1, "correct" means the agent abstained);
**faithfulness** is the judged fraction of answer claims supported by the
retrieved context and vision findings; **vision** marks runs where the agent
routed retrieved figures through the vision model.
