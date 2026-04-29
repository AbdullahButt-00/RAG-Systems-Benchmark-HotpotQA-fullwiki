# RAG Systems Benchmark — HotpotQA FullWiki

A comparative research benchmark of four Retrieval-Augmented Generation (RAG) pipeline architectures evaluated on **HotpotQA (fullwiki)** — a multi-hop question answering dataset requiring evidence assembly from multiple Wikipedia passages.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Pipelines at a Glance](#pipelines-at-a-glance)
- [Evaluation Results](#evaluation-results)
- [Setup](#setup)
- [Running Each Pipeline](#running-each-pipeline)
  - [Dense_RAG](#dense_rag-baseline)
  - [Graph_RAG](#graph_rag)
  - [Hybrid_RAG](#hybrid_rag)
  - [PageIndex_RAG](#pageindex_rag)
- [Metrics Reference](#metrics-reference)
- [Key Findings](#key-findings)
- [Full Technical Report](#full-technical-report)

---

## Overview

HotpotQA questions require evidence from two or more Wikipedia passages to answer correctly — a structure that challenges single-step dense retrieval. This benchmark evaluates four retrieval strategies under the same dataset, corpus, and evaluation harness:

| Pipeline | Core Idea |
|---|---|
| **Dense_RAG** | FAISS flat L2 nearest-neighbor retrieval — strong single-hop baseline |
| **Graph_RAG** | Dense seeds + BFS expansion over a passage-level knowledge graph |
| **Hybrid_RAG** | Confidence-gated staging: graph expansion fires only when dense retrieval is uncertain |
| **PageIndex_RAG** | Cloud-native: keyword pre-filter → PageIndex API handles retrieval and generation |

All pipelines share:
- A deduplicated corpus of **~508,000 passages** built from HotpotQA train + validation splits
- The embedding model `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (384-dim) for local pipelines
- Evaluation on **500 sampled validation questions** (default)
- Identical EM, Token F1, SP-F1, Recall@k, and MRR metric implementations

---

## Repository Structure

```
RAG-Systems-Benchmark-HotpotQA-fullwiki/
│
├── Dense_RAG/                  # Baseline dense retrieval pipeline
│   ├── config.py
│   ├── dataset_download.py     # One-time HotpotQA dataset download
│   ├── preprocess.py           # Builds corpus.json + FAISS index
│   ├── retriever.py            # LlamaIndex VectorStoreIndex + FAISS
│   ├── reader.py               # Ollama local LLM reader
│   ├── pipeline.py             # DenseRAGPipeline (retrieve → read)
│   ├── evaluate.py             # EM, Token F1, SP-F1, Recall@5, MRR, Latency
│   ├── run_evaluation.py       # Entry point
│   ├── evaluation_report.json  # Results (500 samples)
│   ├── requirements.txt
│   ├── data/corpus.json
│   └── storage/                # LlamaIndex docstore + index store
│
├── Graph_RAG/                  # Graph-guided multi-hop pipeline
│   ├── config.py
│   ├── preprocess.py           # Builds corpus.json + FAISS + context_clusters.json
│   ├── graph_builder.py        # NetworkX DiGraph (title_link, entity_overlap, same_context)
│   ├── retriever.py
│   ├── graph_retriever.py      # BFS expansion over passage graph
│   ├── reader.py               # Groq LLM reader
│   ├── pipeline.py             # GraphRAGPipeline (seed → expand → read)
│   ├── evaluate.py             # + Chain Recall@10, Support Coverage
│   ├── run_evaluation.py
│   ├── evaluation_report.json  # Results (500 samples)
│   ├── requirements.txt
│   ├── data/
│   │   ├── corpus.json
│   │   └── context_clusters.json
│   └── storage/
│
├── Hybrid_RAG/                 # Confidence-gated staged hybrid pipeline
│   ├── config.py
│   ├── preprocess.py
│   ├── build_index.py          # One-shot artifact builder
│   ├── retriever.py
│   ├── graph_builder.py        # Leakage-free graph (no same_context edges)
│   ├── graph_retriever.py
│   ├── query_analyzer.py       # Confidence gate (max score + score spread)
│   ├── reranker.py             # Min-max normalize + weighted score fusion
│   ├── reader.py               # Groq LLM reader (temp=0.0, max_tokens=128)
│   ├── pipeline.py             # HybridRAGPipeline (5 stages)
│   ├── evaluate.py             # + Gate Firing Rate (9 metrics total)
│   ├── run_evaluation.py       # Supports ablation CLI flags
│   ├── demo.py                 # Interactive single-question runner
│   ├── evaluation_report.json  # Results (500 samples)
│   ├── requirements.txt
│   └── storage/
│
├── PageIndex_RAG/              # Cloud API-driven pipeline
│   ├── config.py
│   ├── preprocess.py           # Builds corpus + passage bundles (50 passages/doc)
│   ├── ingestor.py             # Uploads bundles to PageIndex, persists doc_ids
│   ├── pageindex_retriever.py  # Keyword pre-filter + PageIndex chat_completions
│   ├── pipeline.py             # PageIndexRAGPipeline
│   ├── evaluate.py
│   ├── run_evaluation.py
│   ├── requirements.txt
│   ├── data/
│   │   ├── corpus.json
│   │   └── passage_lookup.json
│   └── pageindex_index/
│       └── doc_ids.json        # bundle_filename → PageIndex doc_id
│
├── Report.md                   # Full technical analysis report
├── PROJECT_CONTEXT.md          # Project goals, status, and operational runbook
└── .gitignore
```

---

## Dataset

**HotpotQA (fullwiki configuration)**

| Property | Detail |
|---|---|
| Source | `hotpotqa/hotpot_qa` on HuggingFace |
| Configuration | `fullwiki` (~1.3 GB download) |
| Train split | ~90,564 questions |
| Validation split | 7,405 questions |
| Corpus size | ~508,000 deduplicated passages |
| Question types | Bridge (two-hop chain) and Comparison |
| Gold labels | Answer string + supporting fact (title, sentence_index) pairs |

### Downloading the Dataset

Run once from the `Dense_RAG/` directory (the downloaded cache is shared by all pipelines):

```bash
cd Dense_RAG
python dataset_download.py
```

This downloads HotpotQA to the HuggingFace local cache. All pipelines read directly from Arrow files via the `DATASET/` symlink or local cache path configured in each `config.py`.

---

## Pipelines at a Glance

### Dense_RAG — Baseline

```
Question → [FAISS top-5] → [Ollama reader] → Answer
```

- Single-step dense retrieval, no graph
- Reader: local Ollama (`llama3`, CPU inference)
- Build time: ~5–15 min (FAISS index over 508k passages)
- Index size: ~10 GB (FAISS + docstore)

### Graph_RAG — Multi-hop Graph Expansion

```
Question → [FAISS top-5 seeds] → [BFS expansion, max 2 hops] → top-10 → [Groq 70B] → Answer
```

- Three edge types: `title_link` (weight=1.0), `entity_overlap` (weight∝overlap), `same_context` (weight=0.5)
- Hop scoring: `edge_weight / (hop + 2)`
- Reader: Groq `llama-3.3-70b-versatile`
- Build time: ~30–60 min (FAISS + graph)
- Index size: ~15 GB

### Hybrid_RAG — Confidence-Gated Staged

```
Question
  → [FAISS top-5 seeds]
  → [Confidence gate: max_score < 0.40 OR spread < 0.05]
  ├── gate=False: dense-only path
  └── gate=True:  [BFS expansion, max 2 hops, top-10]
  → [EvidenceReranker: 0.6×dense + 0.4×graph, top-7]
  → [Groq 70B]
  → Answer
```

- Leakage-free graph: excludes `same_context` edges
- Gate fired on 10.8% of queries (all due to low score spread)
- Reader: Groq `llama-3.3-70b-versatile`, temp=0.0
- Build time: ~75 min (full) or ~45 min (reusing Dense_RAG FAISS)

### PageIndex_RAG — Cloud API-Driven

```
[Startup] passage_lookup.json → keyword inverted index (title tokens → doc_ids)

Question
  → [Tokenize + stopword filter]
  → [Keyword scoring → top-10 candidate doc_ids]
  → [PageIndex chat_completions per doc_id, citations=True]
  → [Merge: pick response with most citations]
  → [Parse citations → {passage_id, title, text}]
  → Answer + cited passages
```

- No local vector index or graph — ~100 MB footprint
- ~508k passages bundled into ~10,160 documents (50 passages/doc)
- Pre-filter reduces API calls from ~10,160 → ≤10 per query
- Evaluation has not been run

---

## Evaluation Results

All results on **500 HotpotQA validation samples**.

### Primary Metrics

| Metric | Dense_RAG | Graph_RAG | Hybrid_RAG | PageIndex_RAG |
|---|---|---|---|---|
| **Exact Match (EM)** | 0.084 | 0.070 | **0.106** | — |
| **Token F1** | 0.1452 | 0.1256 | **0.1785** | — |
| **Supporting Fact F1** | 0.0509 | 0.0320 | 0.0507 | — |
| **Recall@5** | 0.162 | 0.152 | 0.162 | — |
| **MRR** | **0.1463** | 0.0403 | 0.0403 | — |

### Graph-Specific Metrics

| Metric | Graph_RAG | Hybrid_RAG |
|---|---|---|
| Chain Recall@10 | 0.026 | 0.016 |
| Support Coverage | 0.0988 | 0.000 * |
| Gate Firing Rate | — | 0.108 |

\* *Likely a metric artifact — see [Report.md](Report.md) Section 2.3 for analysis.*

### Latency

| | Dense_RAG | Graph_RAG | Hybrid_RAG |
|---|---|---|---|
| **Mean (ms)** | 48,206 | 89,676 | **3,925** |
| **P95 (ms)** | 80,049 | 130,917 | **7,224** |
| Reader backend | Ollama (local CPU) | Groq API | Groq API |

> **Note:** Latency comparisons are partially confounded by different reader backends. Dense_RAG uses local Ollama (slow CPU inference); Graph_RAG and Hybrid_RAG use the Groq cloud API.

---

## Setup

### Prerequisites

- Python 3.10+
- CPU with ≥16 GB RAM (graph build requires more)
- Active API key for Groq (Graph_RAG, Hybrid_RAG) or PageIndex (PageIndex_RAG)

### 1 — Download the dataset (once, shared by all pipelines)

```bash
cd Dense_RAG
pip install datasets
python dataset_download.py
```

### 2 — Create a virtual environment per pipeline

Each pipeline has its own `requirements.txt`. It is recommended to use separate virtual environments to avoid dependency conflicts.

```bash
# Example for Dense_RAG
cd Dense_RAG
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Repeat the same pattern inside `Graph_RAG/`, `Hybrid_RAG/`, and `PageIndex_RAG/`.

### 3 — Set API keys

Open `config.py` in each pipeline directory and replace the placeholder value:

```python
# Graph_RAG/config.py and Hybrid_RAG/config.py
GROQ_API_KEY = "your_groq_key_here"     # https://console.groq.com

# PageIndex_RAG/config.py
PAGEINDEX_API_KEY = "your_key_here"
```

> Keep keys out of version control. The `.gitignore` does not exclude `config.py` — move keys to environment variables before committing.

---

## Running Each Pipeline

---

### Dense_RAG (Baseline)

**Step 1 — Build corpus and FAISS index** *(run once, ~5–15 min)*

```bash
cd Dense_RAG
source venv/bin/activate
python preprocess.py
```

Creates:
- `data/corpus.json` — 508k deduplicated passages
- `storage/` — LlamaIndex docstore + index metadata
- `faiss_index/index.faiss` — binary FAISS flat L2 index

**Step 2 — Run evaluation**

```bash
python run_evaluation.py                  # 500 samples (default)
python run_evaluation.py --samples 100    # quick smoke test
python run_evaluation.py --samples 7405   # full validation set
```

Output saved to `evaluation_report.json`.

---

### Graph_RAG

**Step 1 — Build corpus and FAISS index** *(~15 min)*

```bash
cd Graph_RAG
source venv/bin/activate
python preprocess.py
```

Creates `data/corpus.json`, `data/context_clusters.json`, `storage/`, and `faiss_index/index.faiss`.

**Step 2 — Build the passage graph** *(~15–45 min)*

```bash
python graph_builder.py
```

Creates `graph_index/graph.pkl` — a serialized NetworkX DiGraph with three edge types:

| Edge | Weight | Source |
|---|---|---|
| `title_link` | 1.0 | Passage P's text mentions passage Q's title |
| `entity_overlap` | min(1.0, shared/5) | Shared capitalised multi-word phrases |
| `same_context` | 0.5 | Co-appeared in same HotpotQA context field |

**Step 3 — Run evaluation**

```bash
python run_evaluation.py                  # 500 samples
python run_evaluation.py --samples 100
python run_evaluation.py --samples 7405
```

---

### Hybrid_RAG

**Option A — Full build from scratch** *(~75 min)*

```bash
cd Hybrid_RAG
source venv/bin/activate
python build_index.py
```

**Option B — Reuse Dense_RAG/Graph_RAG FAISS artifacts** *(~45–60 min, graph only)*

```bash
mkdir -p data faiss_index
ln -s ../../Graph_RAG/data/corpus.json data/corpus.json
cp -r ../../Graph_RAG/storage           storage
cp -r ../../Graph_RAG/faiss_index       faiss_index
python build_index.py --skip_corpus --skip_faiss
```

> The Hybrid graph **must be rebuilt** — it excludes `same_context` edges that are present in Graph_RAG's `graph.pkl`. Using Graph_RAG's graph introduces dataset-leakage risk.

**Run evaluation**

```bash
python run_evaluation.py                                       # default config
python run_evaluation.py --samples 50                          # smoke test
python run_evaluation.py --samples 7405                        # full set
```

**Ablation runs via CLI flags**

| Ablation | Command |
|---|---|
| Always expand (gate always fires) | `--conf_threshold 0.0 --spread_threshold 0.0` |
| Never expand (dense-only) | `--conf_threshold 1.1` |
| No reranking bias | `--alpha 0.5` |
| Threshold calibration sweep | vary `--conf_threshold` across runs |

**Interactive demo**

```bash
python demo.py                               # 5 built-in example questions
python demo.py --question "Who directed ..." # single question
python demo.py --verbose                     # show passage scores + source
python demo.py --diagnostic                  # gate threshold analysis on 50 val examples
```

**Gate threshold calibration**

> Before running the full evaluation, run the diagnostic to inspect score distribution and verify thresholds are appropriate for your index:

```bash
python demo.py --diagnostic
```

LlamaIndex's FAISS backend returns **negative L2 distances** (range: ~`[-1.41, 0]`). The default `CONF_THRESHOLD = 0.40` (positive) means the confidence gate branch never fires in this configuration — only the spread gate activates. Update `CONF_THRESHOLD` in `config.py` to a negative value (e.g., `-0.70`) after running the diagnostic.

---

### PageIndex_RAG

**Step 1 — Build corpus and document bundles** *(~5–10 min)*

```bash
cd PageIndex_RAG
source venv/bin/activate
pip install -r requirements.txt
python preprocess.py
```

Creates:
- `data/corpus.json` — 508k passages
- `data/passage_lookup.json` — passage_id → {title, text, bundle_file, doc_id}
- `pageindex_docs/` — ~10,160 plain-text bundle files (250 passages each)

**Step 2 — Upload bundles to PageIndex** *(time depends on API throughput)*

```bash
python ingestor.py
```

Creates `pageindex_index/doc_ids.json`. The upload is **resumable**: re-running skips already-uploaded bundles.

**Step 3 — Run evaluation**

```bash
python run_evaluation.py                  # 500 samples
python run_evaluation.py --samples 100
python run_evaluation.py --samples 7405
```

---

## Metrics Reference

All metrics are computed in each pipeline's `evaluate.py`. Answer normalization: lowercase → remove articles (a/an/the) → strip punctuation → collapse whitespace.

| Metric | Description | Pipelines |
|---|---|---|
| **Exact Match (EM)** | `normalize(prediction) == normalize(gold)` | All |
| **Token F1** | Token-level precision/recall/F1 (SQuAD formula) | All |
| **Supporting Fact F1 (SP-F1)** | Title-level F1 between retrieved passage titles and gold supporting fact titles | All |
| **Recall@5** | ≥1 gold title in top-5 retrieved | All |
| **Recall@7** | ≥1 gold title in top-7 retrieved | Hybrid_RAG |
| **MRR** | Mean reciprocal rank of first retrieved passage matching a gold title | All |
| **Chain Recall@10** | All gold titles present in top-10 retrieved (full multi-hop chain) | Graph_RAG, Hybrid_RAG |
| **Support Coverage** | Sentence-level: gold (title, sent_idx) pairs covered by retrieved passages | Graph_RAG, Hybrid_RAG |
| **Gate Firing Rate** | Fraction of questions where graph expansion was triggered | Hybrid_RAG |
| **Mean Latency (ms)** | Average wall-clock time per question (retrieval + generation) | All |
| **P95 Latency (ms)** | 95th percentile latency | All |

---

## Key Findings

**1. Hybrid_RAG achieves the best answer quality (EM=10.6%, Token F1=17.85%) and lowest latency (3.9 sec/question).**
Selective graph expansion (10.8% gate rate) avoids the noise penalty of blanket BFS while preserving the ability to handle ambiguous multi-hop queries.

**2. Graph_RAG underperforms Dense_RAG on every metric despite higher complexity.**
BFS expansion introduces noise through imprecise entity-overlap edges. EM drops from 8.4% (Dense) to 7.0% (Graph), and MRR collapses from 0.1463 to 0.0403. The 2× latency increase compounds the disadvantage.

**3. Dense_RAG has the highest MRR (0.1463) — graph augmentation degrades rank precision.**
When dense retrieval finds a relevant passage, it places it near rank 1. Graph expansion and reranking push gold passages down in the ranked list even when they were retrieved correctly as seeds.

**4. The fundamental ceiling is Recall@5 / Recall@7 (≤16.2%).**
Across all pipelines, fewer than 1 in 5 questions have any gold supporting fact in the retrieved set. No architectural layer (graph, gate, reranker) compensates for retrieval failing to include the needed evidence at all.

**5. The Hybrid_RAG confidence gate is miscalibrated in the reported run.**
`CONF_THRESHOLD=0.40` (positive) is incompatible with negative L2 distance scores. All 54 gate firings were triggered exclusively by low score spread, not low confidence. The diagnostic tool in `demo.py --diagnostic` should be run before any new evaluation to set an appropriate negative threshold.

**6. PageIndex_RAG has not been evaluated.**
The pipeline is fully implemented and bundles are uploaded (doc_ids.json exists), but no evaluation results exist. Running `run_evaluation.py` requires a valid PageIndex API key.

---

## Full Technical Report

See [Report.md](Report.md) for the complete technical analysis, including:

- Deep per-pipeline architecture breakdown
- Source code–level methodology descriptions
- Cross-pipeline comparative analysis with insights
- 8 concrete recommendations for improving system performance
