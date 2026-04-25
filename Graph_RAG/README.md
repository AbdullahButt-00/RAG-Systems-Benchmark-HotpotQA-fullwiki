# Graph-RAG Pipeline — HotpotQA fullwiki

Graph-guided Retrieval-Augmented Generation for multi-hop QA, extending
Dense-RAG with a NetworkX passage graph for multi-hop evidence assembly.

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │          GraphRAGPipeline            │
                        └─────────────────────────────────────┘
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │                  GraphRetriever                       │
              │                                                       │
              │  Question ──► [FAISS Dense Retrieval] ──► 5 seeds    │
              │                         │                             │
              │               [NetworkX BFS Expansion]               │
              │                    max 2 hops                        │
              │                         │                             │
              │               top-10 passages (ranked)               │
              └──────────────────────────┬──────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   Groq LLM Reader    │
                              │  llama-3.3-70b-vers. │
                              └──────────┬──────────┘
                                         │
                                       Answer
```

## How Graph-RAG Differs from Dense-RAG

| Aspect | Dense-RAG | Graph-RAG |
|---|---|---|
| Retrieval | FAISS top-5 only | FAISS seeds + graph BFS expansion |
| Passages returned | 5 | up to 10 |
| Multi-hop coverage | Limited | Explicit — follows passage links |
| Graph edges | None | title_link, entity_overlap, same_context |
| Extra metrics | — | Chain Recall@10, Support Coverage |
| Build time | ~15 min | ~30–60 min (graph build adds ~15–45 min) |

---

## Setup

### 1. Create and activate virtual environment

```bash
cd /home/abdullah/Desktop/AGENTIC/AGENTIC_RESEARCH/Graph_RAG
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Set your Groq API key

Open `config.py` and confirm / update:

```python
GROQ_API_KEY = "your_key_here"
```

---

## Build Pipeline (run once)

### Step 1 — Build corpus + FAISS index

```bash
python preprocess.py
```

Creates:
- `data/corpus.json`            — 508k+ deduplicated passages with sentences
- `data/context_clusters.json`  — passage co-occurrence clusters per question
- `storage/`                    — LlamaIndex docstore
- `faiss_index/index.faiss`     — FAISS flat L2 index

Expected time: ~15 min (embedding 508k passages on CPU)

### Step 2 — Build the passage graph

```bash
python graph_builder.py
```

Creates:
- `graph_index/graph.pkl`  — serialised NetworkX DiGraph

Three edge types are built:
1. **title_link** (`weight=1.0`) — passage P's text mentions passage Q's title
2. **entity_overlap** (`weight∝overlap`) — passages share capitalised multi-word phrases
3. **same_context** (`weight=0.5`) — passages appeared together in same HotpotQA question

Expected time: 15–45 min depending on corpus size.

### Step 3 — Run evaluation

```bash
python run_evaluation.py                 # 500 samples (default)
python run_evaluation.py --samples 100   # quick test
python run_evaluation.py --samples 7405  # full validation set
```

---

## Expected Output

```
============================================================
Graph-RAG Evaluation Report — HotpotQA fullwiki
============================================================
Samples Evaluated      : 500
------------------------------------------------------------
Exact Match (EM)       : 0.XXXX
Token F1               : 0.XXXX
Supporting Fact F1     : 0.XXXX
Recall@5               : 0.XXXX
MRR                    : 0.XXXX
Chain Recall@10        : 0.XXXX
Support Coverage       : 0.XXXX
Mean Latency (ms)      : XXX.X
P95 Latency  (ms)      : XXX.X
============================================================
```

Full per-question breakdown saved to `evaluation_report.json`.

---

## Metrics Explained

| # | Metric | What it measures |
|---|---|---|
| 1 | **Exact Match (EM)** | Predicted answer == gold answer after normalisation |
| 2 | **Token F1** | Token-level overlap between predicted and gold answer |
| 3 | **Supporting Fact F1** | Title-level F1 between retrieved and gold supporting docs |
| 4 | **Recall@5** | ≥1 gold title in top-5 retrieved passages |
| 5 | **MRR** | Mean reciprocal rank of first gold-title match |
| 6 | **Chain Recall@10** | ALL gold titles present in top-10 retrieved (full chain assembled) |
| 7 | **Support Coverage** | Sentence-level: gold (title, sent_idx) pairs covered by retrieval |
| 8 | **Latency** | Mean and P95 wall-clock time per question (ms) |

Chain Recall@10 and Support Coverage are the Graph-RAG-specific metrics that
reward assembling the complete multi-hop evidence chain, not just finding
one supporting document.

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `EMBED_MODEL_NAME` | `multi-qa-MiniLM-L6-cos-v1` | QA-optimised sentence encoder |
| `TOP_K_SEED` | `5` | Dense seeds before graph expansion |
| `TOP_K_GRAPH` | `10` | Passages returned after expansion |
| `MAX_HOP` | `2` | BFS depth limit |
| `SHARED_ENTITY_THRESHOLD` | `1` | Min shared phrases for entity edge |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model for answer generation |
| `EVAL_SAMPLE_SIZE` | `500` | Default validation samples |
