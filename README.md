# nim-multimodal-agent

[![tests](https://github.com/Karthikvenugopal/nim-multimodal-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Karthikvenugopal/nim-multimodal-agent/actions/workflows/ci.yml)

A **multimodal agentic RAG pipeline** built on [NVIDIA NIM](https://build.nvidia.com)
with a benchmark-style evaluation layer. A LangGraph agent retrieves over a mixed
text + image corpus, routes retrieved figures through a NIM vision-language model
to extract the facts they contain, and generates grounded answers with a Nemotron
text model. A **faithfulness-gated self-correction loop** (Corrective-RAG /
Self-RAG / Reflexion) turns the LLM-as-judge into an *in-graph control signal*:
the agent judges its own answer's grounding and, if it falls short, takes one
corrective action and retries. The evaluation layer measures the loop and runs a
vision **ablation** that quantifies how much the multimodal path contributes.

<p align="center">
  <img src="docs/graph.png" alt="LangGraph agent with self-correction loop: retrieve → generate → judge_gate → (correct → generate | abstain | accept)" width="220">
</p>

- **Genuinely multimodal**: the figure data in `corpus/images/` (latency
  benchmarks, revenue mix, GPU utilization, pipeline diagram, error rates)
  exists *only in the pixels* — figure and cross-modal questions cannot be
  answered from the text, so they exercise the real vision path
  (base64 image → NIM VLM).
- **Genuinely agentic**: a compiled LangGraph `StateGraph` with a conditional
  edge that routes to vision analysis only when a retrieved image passes a
  relevance gate (top-ranked, or above an absolute similarity threshold), so
  the vision model fires on figure/cross-modal questions and is skipped on
  pure-text ones.
- **Cross-modal reasoning**: three benchmark questions (`CM*`) require *fusing*
  a fact read from a figure with a fact retrieved from text — e.g. "which
  variant has the lowest p95 latency (figure), and how many camera streams does
  it support (text)?" Neither modality answers them alone.
- **Self-correcting**: the faithfulness judge runs *inside* the graph as a gate
  after generation. Below-threshold answers trigger one corrective action —
  force the vision tool on an unread figure, rewrite the query and re-retrieve,
  or decompose a multi-part question — then regenerate and re-judge, abstaining
  honestly after `MAX_CORRECTIONS` rather than returning a low-faith answer.
- **Honest evaluation**: a 14-question labeled set (text-answerable, figure-only,
  cross-modal, and one unanswerable question that tests abstention), scored for
  answer correctness and context-faithfulness by a Nemotron judge — with a
  vision ablation that reruns the set blind to measure the multimodal lift and a
  correction harness that reports trigger/recovery rates and before/after grounding.

## Pipeline

```
retrieve ──(relevant image queued?)──> analyze_images ──> generate
    └──────────(text only)──────────────────────────────────^
```

1. **Ingest** — text docs are split into passage chunks; image captions index
   the figures. All chunks are embedded with a NIM retrieval model.
2. **Retrieve** — top-k chunks by cosine similarity. A retrieved image is queued
   for vision only if it is rank-1 or scores above a similarity threshold.
3. **Analyze images** — each queued figure is sent (base64) to the NIM
   vision-language model, which extracts the labels/values it shows.
4. **Generate** — the Nemotron text model answers *only* from the text passages
   plus the vision findings, abstaining when the context is insufficient.
5. **Judge gate** (correction mode) — the answer's grounding is scored in-graph;
   if it clears `FAITHFULNESS_THRESHOLD` the agent returns it, otherwise it
   corrects and retries (see below).

```
... generate ──> judge_gate ──(grounded?)──────────────> answer
                     │  └──(low faith, tries left)──> correct ──┐
                     │                                          │
                     └──(low faith, out of tries)──> abstain    │
                              ^                                  │
                              └──────── generate <──────────────┘
```

## Models (NVIDIA NIM, OpenAI-compatible API)

| Role | Default model | Env var |
|---|---|---|
| Vision-language | `nvidia/nemotron-nano-12b-v2-vl` | `NIM_VISION_MODEL` |
| Generation + judge | `nvidia/llama-3.3-nemotron-super-49b-v1.5` | `NIM_TEXT_MODEL` |
| Retrieval embeddings | `nvidia/llama-nemotron-embed-1b-v2` | `NIM_EMBED_MODEL` |

All calls go through `https://integrate.api.nvidia.com/v1` via the `openai`
SDK, with retry/backoff on transient errors. The NIM catalog changes often —
override any model with the env vars above (see `.env.example`).

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

# text-only baseline (vision path disabled)
python main.py --benchmark --no-vision

# ablation: run the benchmark with and without vision, report the lift
python main.py --ablation

# self-correction loop: run the benchmark and report loop metrics
python main.py --correction

# single question through the correction loop (prints the loop trace)
python main.py "What is the p95 latency of the VoltEdge Max?" --correction
```

## Repo layout

```
corpus/docs/            3 markdown docs (text facts)
corpus/images/          5 chart/diagram PNGs (figure-only facts)
corpus/manifest.json    image captions used for retrieval (values withheld)
corpus/questions.json   14 labeled questions (text / figure / cross_modal / unanswerable)
nim_client.py           NIM chat / vision / embedding / query-rewrite (+ retry/backoff)
judge.py                shared LLM-as-judge primitives (benchmark + in-graph gate)
agent.py                LangGraph graph: retrieve → route → vision → generate → correction loop
evaluate.py             benchmark harness, LLM-as-judge, vision ablation, correction metrics
main.py                 CLI (--benchmark / --ablation / --correction)
scripts/make_images.py  regenerates the corpus figures (matplotlib)
scripts/render_graph.py renders docs/graph.png from the compiled graph
tests/test_offline.py   offline unit tests (no API key needed); run via pytest
docs/graph.png          rendered LangGraph diagram (incl. correction loop)
```

## Benchmark results

Real output of `python main.py --benchmark` against the live NIM API:

```
models: vision=nvidia/nemotron-nano-12b-v2-vl  text=nvidia/llama-3.3-nemotron-super-49b-v1.5  embed=nvidia/llama-nemotron-embed-1b-v2
ingesting corpus...
ingested 16 chunks (11 text, 5 image)

  [T1] PASS  vision=n  faith=1.00  (3.5s)
  [T2] PASS  vision=n  faith=1.00  (5.7s)
  [T3] PASS  vision=n  faith=1.00  (28.7s)
  [T4] PASS  vision=n  faith=1.00  (4.3s)
  [T5] PASS  vision=n  faith=1.00  (7.4s)
  [F1] PASS  vision=Y  faith=1.00  (14.2s)
  [F2] PASS  vision=Y  faith=1.00  (25.3s)
  [F3] PASS  vision=Y  faith=1.00  (27.1s)
  [F4] PASS  vision=Y  faith=1.00  (8.3s)
  [F5] PASS  vision=Y  faith=1.00  (24.6s)
  [U1] PASS  vision=n  faith=1.00  (2.2s)
  [CM1] PASS  vision=Y  faith=1.00  (14.2s)
  [CM2] PASS  vision=Y  faith=1.00  (21.5s)
  [CM3] PASS  vision=Y  faith=1.00  (15.2s)

id    type          vision  correct  faithful    sec
----------------------------------------------------
T1    text          no      1        1.00        3.5
T2    text          no      1        1.00        5.7
T3    text          no      1        1.00       28.7
T4    text          no      1        1.00        4.3
T5    text          no      1        1.00        7.4
F1    figure        yes     1        1.00       14.2
F2    figure        yes     1        1.00       25.3
F3    figure        yes     1        1.00       27.1
F4    figure        yes     1        1.00        8.3
F5    figure        yes     1        1.00       24.6
U1    unanswerable  no      1        1.00        2.2
CM1   cross_modal   yes     1        1.00       14.2
CM2   cross_modal   yes     1        1.00       21.5
CM3   cross_modal   yes     1        1.00       15.2
----------------------------------------------------

type            n  accuracy  vision-fire
cross_modal     3   100.0%      100.0%
figure          5   100.0%      100.0%
text            5   100.0%        0.0%
unanswerable    1   100.0%        0.0%

questions:                     14
answer accuracy:               100.0%
mean faithfulness:             1.00
vision-dependent accuracy:     100.0%  (figure + cross_modal)
vision fired when needed:      100.0%
```

Metric notes: **correct** is judged against the gold label (for the
unanswerable question U1, "correct" means the agent abstained);
**faithfulness** is the judged fraction of answer claims supported by the
retrieved context and vision findings; **vision** marks runs where the agent
routed retrieved figures through the vision model.

## Vision ablation

To show the multimodal path is doing real work — not that the text model is
guessing figure values — `python main.py --ablation` reruns the whole set with
the vision route disabled. Figure and cross-modal questions then lose the only
source of their answer. Real output against the live NIM API:

```
=== ablation: answer accuracy, vision ON vs OFF ===
type            n  vision-on  vision-off    delta
-------------------------------------------------
cross_modal     3    100.0%      66.7%   +33.3
figure          5    100.0%      40.0%   +60.0
text            5    100.0%     100.0%    +0.0
unanswerable    1    100.0%     100.0%    +0.0
-------------------------------------------------
overall        14    100.0%      71.4%   +28.6
```

Disabling vision leaves text and unanswerable questions untouched (+0.0) but
collapses figure-only accuracy (100% → 40%) and cross-modal accuracy
(100% → 66.7%), for a **+28.6-point overall lift** from the multimodal path.
The figure drop is not a clean 100% → 0%: some figure questions remain
answerable blind because the text model produces a plausible answer the judge
accepts — a useful reminder that an LLM judge can over-credit guessable
questions, and exactly the kind of signal an ablation is meant to expose.

> **Variance caveat (honest):** the vision-**on** side is stable at 100%, but
> the vision-**off** baseline is noisy run-to-run because it depends on the text
> model guessing and the judge's leniency. A second run of the same command gave
> overall 85.7% blind (figure 100%, cross-modal 33%). The takeaway is
> directional — vision reliably grounds these answers; text-only does not — and
> with a 14-question corpus the exact blind percentage should not be
> over-read.

## Self-correction loop

`python main.py --correction` turns the faithfulness judge into an in-graph
control signal and measures it. So the loop is the thing that grounds figure
answers (and is therefore measurable), correction mode is **text-first**: the
agent answers from text, self-checks grounding, and escalates to the vision
tool / query rewrite / decomposition only when the gate fails — a cheap-first
Corrective-RAG policy. An answer that abstains grounds nothing, so it scores 0
at the gate and must be corrected; abstaining is the honest final move only once
`MAX_CORRECTIONS` is reached. Real output against the live NIM API:

```
id    type          att actions                  pre  post result   correct
---------------------------------------------------------------------------
T1    text            0 -                       1.00  1.00 accept         1
T2    text            0 -                       1.00  1.00 accept         1
T3    text            0 -                       1.00  1.00 accept         1
T4    text            0 -                       1.00  1.00 accept         1
T5    text            0 -                       1.00  1.00 accept         1
F1    figure          1 force_vision            0.00  1.00 recover        1
F2    figure          1 force_vision            0.00  1.00 recover        1
F3    figure          1 force_vision            0.00  1.00 recover        1
F4    figure          1 force_vision            0.00  1.00 recover        1
F5    figure          1 force_vision            0.00  1.00 recover        1
U1    unanswerable    2 force_vision,rewrite_query  0.00  0.00 abstain        1
CM1   cross_modal     1 force_vision            0.00  1.00 recover        1
CM2   cross_modal     1 force_vision            0.00  1.00 recover        1
CM3   cross_modal     1 force_vision            0.00  1.00 recover        1
---------------------------------------------------------------------------

threshold:                   0.70
questions:                   14
correction trigger rate:     9/14 = 64.3%
recovery rate:               8/9 = 88.9%
mean faithfulness pre-corr:  0.00  (triggered only)
mean faithfulness post-corr: 0.89  (triggered only)
abstention rate:             1/14 = 7.1%
final answer accuracy:       14/14 = 100.0%
actions fired:               force_vision=9  rewrite_query=1
```

How to read this:

- **Trigger rate 64.3%** — the five text questions ground from text and pass the
  gate untouched; the nine figure / cross-modal / unanswerable questions cannot
  be grounded from text alone, so the gate catches them.
- **Recovery rate 88.9%** — eight of those nine clear the gate after one
  corrective action (the policy escalates to `force_vision`, which transcribes
  the figure the text-first pass skipped). Final accuracy is 14/14, the same as
  the vision-on benchmark, but reached by self-correction from a blind start.
- **Mean post-correction faithfulness 0.89, not 1.0** — the one unanswerable
  question (`U1`) tries `force_vision` then `rewrite_query`, still cannot ground
  an answer, and **correctly abstains** (it stays at 0.00, pulling the triggered
  average down). The loop knowing when to give up is the point, not a failure.

**Honest caveats.** The corpus is small (14 questions), so these rates are
illustrative, not statistically robust. The setup is deliberately text-first to
*exercise* the loop; the default pipeline (`--benchmark`) already grounds every
answer in one pass, so the loop's value here is the mechanism (self-checking and
escalating) and the abstention safety net, demonstrated on a small set, rather
than a large measured accuracy gain over the base pipeline.

## Tests

Offline unit tests (corpus loading, judge-JSON parsing, graph routing and the
relevance gate, the `--no-vision` path, the self-correction policy and gate,
retry/backoff, response cleaning) run without an API key and execute in CI on
every push:

```bash
pip install -r requirements-dev.txt
pytest -q
```
