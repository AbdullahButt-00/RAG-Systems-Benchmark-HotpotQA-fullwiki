# Confidence-Gated Staged Hybrid RAG

Multi-hop QA pipeline for HotpotQA fullwiki combining dense retrieval with
conditional graph expansion, governed by a confidence gate.

---

## Pipeline overview

```
Question
  │
  ▼
Stage 1: DenseRetriever.retrieve_seed(top_k=5)
  │           FAISS IndexFlatL2, multi-qa-MiniLM-L6-cos-v1 (384-dim)
  │
  ▼
Stage 2: QueryAnalyzer.analyze(seeds)
  │           gate_fired = (max_score < CONF_THRESHOLD)
  │                     OR (score_spread < SPREAD_THRESHOLD)
  │
  ├── gate_fired=False ──────────────────────────────────────────────┐
  │                                                                  │
  ▼ gate_fired=True                                                  │
Stage 3: GraphRetriever.expand_from_seeds(seeds, top_k=10, hops=2)  │
  │           BFS over title_link + entity_overlap edges             │
  │           (no same_context edges — leakage-free graph)           │
  │                                                                  │
  ▼                                                                  ▼
Stage 4: EvidenceReranker.rerank(seeds, expanded, top_k=7)
  │           final_score = α * dense_norm + (1-α) * graph_norm
  │           α = RERANK_ALPHA (default 0.6)
  │
  ▼
Stage 5: reader.generate_answer(question, top_7_passages)
  │           Groq llama-3.3-70b-versatile, temp=0.0, max_tokens=128
  │
  ▼
Answer
```

---

## Directory structure

```
Confidence-Gated Staged Hybrid RAG/
├── config.py          — all hyperparameters and paths
├── preprocess.py      — corpus.json + FAISS index builder
├── retriever.py       — DenseRetriever (LlamaIndex + FAISS)
├── graph_builder.py   — leakage-free passage graph builder
├── graph_retriever.py — BFS expansion (expand_from_seeds + retrieve)
├── query_analyzer.py  — confidence gate + threshold diagnostic
├── reranker.py        — score normalization + evidence fusion
├── reader.py          — Groq LLM reader
├── pipeline.py        — HybridRAGPipeline (full end-to-end)
├── evaluate.py        — all 9 metrics including gate_firing_rate
├── run_evaluation.py  — main evaluation script (500-sample default)
├── build_index.py     — one-shot artifact builder
├── demo.py            — interactive single-question demo
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Build artifacts

### Option A — full build from scratch (~75 min total)

```bash
cd "Confidence-Gated Staged Hybrid RAG"
python build_index.py
```

### Option B — reuse Graph_RAG artifacts (corpus + FAISS identical)

```bash
cd "Confidence-Gated Staged Hybrid RAG"
mkdir -p data faiss_index
ln -s ../../Graph_RAG/data/corpus.json  data/corpus.json
cp -r ../../Graph_RAG/storage           storage
cp -r ../../Graph_RAG/faiss_index       faiss_index
python build_index.py --skip_corpus --skip_faiss   # only builds graph (~45–60 min)
```

**The graph MUST be rebuilt.** The hybrid graph excludes `same_context` edges
that are present in Graph_RAG's `graph.pkl`. Using Graph_RAG's graph would
re-introduce the dataset-leakage risk.

---

## Run evaluation

```bash
python run_evaluation.py                      # 500 samples, default config
python run_evaluation.py --samples 50         # quick smoke test
python run_evaluation.py --samples 7405       # full validation set
```

### Ablation runs via CLI flags

| Ablation | Command |
|---|---|
| A2: always expand | `--conf_threshold 0.0 --spread_threshold 0.0` |
| A3: never expand  | `--conf_threshold 1.1` |
| A4: no reranking bias | `--alpha 0.5` |
| A7: threshold sweep | run multiple times varying `--conf_threshold` |

---

## Interactive demo

```bash
python demo.py                                # 5 example questions
python demo.py --question "Who directed ..."  # single question
python demo.py --verbose                      # show passage scores + sources
python demo.py --diagnostic                   # threshold diagnostic on 50 val examples
```

---

## Key configuration (config.py)

| Parameter | Default | Meaning |
|---|---|---|
| `CONF_THRESHOLD` | 0.40 | max seed score below which gate fires |
| `SPREAD_THRESHOLD` | 0.05 | score spread below which gate fires |
| `RERANK_ALPHA` | 0.6 | weight for dense scores in reranker |
| `TOP_K_SEED` | 5 | dense seeds retrieved |
| `TOP_K_GRAPH` | 10 | max BFS-expanded passages |
| `TOP_K_FINAL` | 7 | passages sent to LLM reader |
| `MAX_HOP` | 2 | BFS depth limit |

---

## Metrics

| Metric | Type | Notes |
|---|---|---|
| EM | Primary | Exact match after normalization |
| Token F1 | Primary | Token overlap |
| SP-F1 | Primary | Supporting fact title-level F1 |
| Recall@5 | Primary | Baseline-compatible |
| Recall@7 | Primary | Hybrid-specific (TOP_K_FINAL) |
| MRR | Primary | Mean reciprocal rank |
| Chain Recall | Primary | All gold titles retrieved (multi-hop) |
| Support Coverage | Primary | Sentence-level coverage |
| Gate Firing Rate | Diagnostic | % questions where graph expansion fired |
| Latency | Diagnostic | Mean + P95 |

---

## Academic notes

**Why the graph excludes same_context edges:**
Graph_RAG's `graph.pkl` includes `same_context` edges — connections between
passages that co-appeared in the same HotpotQA question's context field. These
edges encode dataset structure (which passages the dataset paired together),
creating a data-leakage risk that inflates retrieval metrics. The hybrid graph
uses only `title_link` and `entity_overlap` edges, which are derived purely
from passage text with no reference to question labels.

To quantify this leakage, run ablation A6 using Graph_RAG's graph.pkl and
compare against this pipeline's `graph_no_context.pkl`.

**Paper claim:** "We propose a staged hybrid retrieval architecture that selectively
triggers knowledge-graph expansion based on dense retrieval confidence, improving
recall on complex multi-hop queries without degrading precision on simple queries."
