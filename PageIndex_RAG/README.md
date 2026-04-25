# PageIndex-RAG Pipeline — HotpotQA fullwiki

PageIndex-Guided Retrieval-Augmented Generation for multi-hop QA.
Replaces the FAISS retriever + Groq LLM reader combination with the
PageIndex API, which handles both document search and answer generation
in a single call.

---

## Architecture

```
                    ┌──────────────────────────────────┐
                    │       PageIndexRAGPipeline        │
                    └──────────────────────────────────┘
                                      │
            ┌─────────────────────────▼──────────────────────────┐
            │               PageIndexRetriever                     │
            │                                                      │
            │  Question ──► [Keyword Pre-filter]                  │
            │                    │                                 │
            │        keyword → doc_id inverted index              │
            │          (built once at startup from                 │
            │           passage_lookup.json titles)               │
            │                    │                                 │
            │         up to 10 candidate doc_ids                  │
            │                    │                                 │
            │  ┌─────────────────▼────────────────────────────┐   │
            │  │  PageIndex chat_completions (per doc_id)      │   │
            │  │  - enable_citations=True                      │   │
            │  │  - temperature=0                              │   │
            │  └─────────────────┬────────────────────────────┘   │
            │                    │                                 │
            │          merge responses; pick best                  │
            │          (most citations) → answer + cited_passages  │
            └─────────────────────┬──────────────────────────────┘
                                  │
                           answer + citations
                           (passage_id, title, text)
```

## How PageIndex-RAG Differs from Dense-RAG and Graph-RAG

| Aspect | Dense-RAG | Graph-RAG | PageIndex-RAG |
|---|---|---|---|
| Retrieval | FAISS top-5 | FAISS + BFS graph | PageIndex API (citations) |
| Generation | Groq LLM | Groq LLM | PageIndex API (built-in) |
| Local index size | ~10 GB (FAISS+docstore) | ~15 GB | ~100 MB (bundle files) |
| Build time | ~15 min | ~45–60 min | ~30 min (upload bundles) |
| Dependencies | faiss, torch, llama-index | + networkx | pageindex, datasets only |
| Extra infra | None | NetworkX graph | PageIndex cloud |

---

## Setup

### 1. Create and activate virtual environment

```bash
cd /home/abdullah/Desktop/AGENTIC/AGENTIC_RESEARCH/PageIndex_RAG
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Set your PageIndex API key

Open `config.py` and set:

```python
PAGEINDEX_API_KEY = "your_key_here"
```

---

## Build Pipeline (run once)

### Step 1 — Build corpus and document bundles

```bash
python preprocess.py
```

Creates:
- `data/corpus.json`          — 508k+ deduplicated passages
- `data/passage_lookup.json`  — passage_id → {title, text, bundle_file, doc_id}
- `pageindex_docs/`           — ~10,160 plain-text bundle files (50 passages each)

Expected time: ~5–10 min (no embedding — pure text processing)

### Step 2 — Upload bundles to PageIndex

```bash
python ingestor.py
```

Creates:
- `pageindex_index/doc_ids.json`  — bundle_filename → doc_id mapping

This step is resumable: if interrupted, re-running skips already-uploaded bundles.
doc_ids are persisted after each successful upload.

Expected time: depends on PageIndex API throughput (~10,160 uploads)

### Step 3 — Run evaluation

```bash
python run_evaluation.py                 # 500 samples (default)
python run_evaluation.py --samples 100   # quick test
python run_evaluation.py --samples 7405  # full validation set
```

---

## Option B Pre-filter Strategy

HotpotQA has ~508k passages bundled into ~10,160 documents.
Querying all documents per question is impractical.

At startup, `PageIndexRetriever` builds an inverted index:

```
title_keyword → [doc_id_1, doc_id_2, ...]
```

using tokens from every passage title in `passage_lookup.json`.

At query time:
1. Question is tokenized (lowercase, strip punctuation, remove stopwords)
2. Each token is looked up in the inverted index
3. Doc_ids are scored by number of keyword hits
4. Top `MAX_CANDIDATE_DOCS` (default: 10) doc_ids are selected
5. Only those doc_ids are queried via `chat_completions`

This reduces per-question API calls from ~10,160 to ≤10.

---

## Expected Output

```
============================================================
PageIndex-RAG Evaluation Report — HotpotQA fullwiki
============================================================
Samples Evaluated        : 500
------------------------------------------------------------
Exact Match (EM)         : 0.XXXX
Token F1                 : 0.XXXX
Supporting Fact F1       : 0.XXXX
Recall@5                 : 0.XXXX
MRR                      : 0.XXXX
Mean Latency (ms)        : XXX.X
P95 Latency  (ms)        : XXX.X
============================================================
```

Full per-question breakdown saved to `evaluation_report.json`.

---

## Metrics

| # | Metric | What it measures |
|---|---|---|
| 1 | **Exact Match (EM)** | Predicted answer == gold answer after normalisation |
| 2 | **Token F1** | Token-level overlap between predicted and gold answer |
| 3 | **Supporting Fact F1** | Title-level F1 between cited passages and gold supporting docs |
| 4 | **Recall@5** | ≥1 gold title in top-5 cited passages |
| 5 | **MRR** | Mean reciprocal rank of first citation matching a gold title |
| 6 | **Latency** | Mean and P95 wall-clock time per question (ms) |

Metrics 1–5 use identical logic to Dense-RAG (`evaluate.py` is copied verbatim).
The only difference: `cited_passages` from PageIndex replaces `retrieved_passages`
from FAISS — both use the same `{passage_id, title, text}` dict shape.

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `PAGEINDEX_API_KEY` | `"your_key_here"` | PageIndex API key |
| `PASSAGES_PER_DOCUMENT` | `50` | Passages bundled per submitted document |
| `MAX_CANDIDATE_DOCS` | `10` | Max doc_ids queried per question (pre-filter cap) |
| `PAGEINDEX_POLL_INTERVAL` | `5` | Seconds between status polls during upload |
| `PAGEINDEX_ENABLE_CITATIONS` | `True` | Request citations from PageIndex |
| `PAGEINDEX_TEMPERATURE` | `0` | Generation temperature |
| `EVAL_SAMPLE_SIZE` | `500` | Default validation samples |
