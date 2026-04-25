# Dense-RAG Pipeline — HotpotQA fullwiki

A retrieval-augmented generation pipeline for multi-hop QA, built with:
- **LlamaIndex** — document ingestion, indexing, retrieval abstraction
- **FAISS** (flat L2) — dense vector store via `llama-index-vector-stores-faiss`
- **sentence-transformers/all-MiniLM-L6-v2** — passage and query encoder
- **Groq (llama3-8b-8192)** — LLM reader

---

## Project Structure

```
AGENTIC_RESEARCH/
├── config.py            # All paths, model names, API keys, hyperparams
├── preprocess.py        # Build corpus.json + FAISS index (run once)
├── retriever.py         # LlamaIndex VectorStoreIndex + FaissVectorStore
├── reader.py            # Groq LLM reader with retry logic
├── pipeline.py          # DenseRAGPipeline: retrieve → read → answer
├── evaluate.py          # EM, Token F1, SP-F1, Recall@5, MRR, Latency
├── run_evaluation.py    # Entry point: evaluate on validation set
├── requirements.txt
├── data/
│   └── corpus.json      # Built by preprocess.py (passage_id → title + text)
├── storage/             # LlamaIndex docstore + index store (built once)
├── faiss_index/
│   └── index.faiss      # Binary FAISS index (built once)
└── evaluation_report.json
```

---

## Setup

### 1. Create and activate virtual environment

```bash
cd /home/abdullah/Desktop/AGENTIC/AGENTIC_RESEARCH
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Set your Groq API key

Open `config.py` and replace the placeholder:

```python
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
```

Get a free key at https://console.groq.com

---

## Usage

### Step 1 — Build the corpus and FAISS index (run once)

```bash
python preprocess.py
```

This will:
1. Load HotpotQA train + validation splits from the local `DATASET/` cache
2. Deduplicate passages by title → save `data/corpus.json`
3. Encode all passages with `all-MiniLM-L6-v2` (batches of 512)
4. Build a `FAISS IndexFlatL2(384)` index via LlamaIndex
5. Persist the index to `storage/` and `faiss_index/index.faiss`

Expected time: ~5–15 min depending on corpus size and CPU. Run only once.

### Step 2 — Run evaluation

```bash
python run_evaluation.py                  # 500 samples (default)
python run_evaluation.py --samples 100    # quick test
python run_evaluation.py --samples 7405   # full validation set
```

### Expected output

```
============================================================
Dense-RAG Evaluation Report — HotpotQA fullwiki
============================================================
Samples evaluated        : 500
Exact Match (EM)         : 0.XXXX
Token F1                 : 0.XXXX
Supporting Fact F1       : 0.XXXX
Recall@5                 : 0.XXXX
MRR                      : 0.XXXX
Mean Latency (ms)        : XXX.X
P95 Latency  (ms)        : XXX.X
============================================================
Full report saved → .../evaluation_report.json
```

Results are also saved to `evaluation_report.json`.

---

## Metrics

| Metric | Description |
|---|---|
| **Exact Match (EM)** | Fraction where normalized prediction == normalized gold answer |
| **Token F1** | Token-level overlap F1 (standard SQuAD formula) |
| **Supporting Fact F1** | Title-level F1 between retrieved passages and gold supporting facts |
| **Recall@5** | Fraction where ≥1 gold supporting fact title appears in top-5 results |
| **MRR** | Mean reciprocal rank of first retrieved passage matching a gold title |
| **Latency** | End-to-end wall-clock time per question (retrieval + generation) |

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `EMBED_MODEL_NAME` | `multi-qa-MiniLM-L6-cos-v1` | Sentence encoder |
| `EMBED_DIM` | `384` | Embedding dimension |
| `EMBED_BATCH_SIZE` | `512` | Encoding batch size |
| `TOP_K` | `5` | Passages retrieved per question |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model ID |
| `EVAL_SAMPLE_SIZE` | `500` | Default validation samples |
