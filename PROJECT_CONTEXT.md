# AGENTIC RAG PIPELINES - Project Context Document

## 1) What this project is

This repository is a comparative RAG research workspace for multi-hop question answering on **HotpotQA (fullwiki)**.

It contains three independent pipeline variants:

- `Dense_RAG`: Dense retrieval with FAISS + LLM answer generation.
- `Graph_RAG`: Dense retrieval plus graph expansion for better multi-hop evidence coverage.
- `PageIndex_RAG`: Cloud-hosted retrieval+generation using PageIndex citations.

The shared objective is to evaluate retrieval and answer quality across different RAG designs under the same dataset and broadly similar metrics.

## 2) Core problem it solves

HotpotQA questions often require evidence from multiple passages. A single-hop retriever can miss parts of the evidence chain.

This project explores three strategies to improve end-to-end QA:

1. **Dense baseline**: retrieve nearest passages in embedding space.
2. **Graph-guided retrieval**: start from dense seeds, then expand through passage links.
3. **Managed retrieval+generation API**: offload retrieval and generation to PageIndex with citations.

## 3) Current implementation status

### `Dense_RAG` (baseline)
- Builds a deduplicated corpus from locally cached HotpotQA splits.
- Encodes passages with `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`.
- Stores vectors in FAISS via LlamaIndex.
- Uses Groq (`llama-3.3-70b-versatile`) as the reader model.
- Evaluates with EM, Token F1, Supporting Fact F1, Recall@5, MRR, latency.

### `Graph_RAG` (multi-hop enhancement)
- Reuses dense retrieval artifacts (corpus + FAISS).
- Builds a `NetworkX` directed graph over passages.
- Edge signals include:
  - title mention links,
  - entity overlap,
  - same-context co-occurrence.
- Retrieval path: dense seed passages -> BFS graph expansion -> reranked candidates.
- Adds graph-oriented evaluation signals: Chain Recall@10 and Support Coverage.

### `PageIndex_RAG` (API-driven variant)
- Creates text bundles from passages and uploads to PageIndex.
- Persists mapping from bundle files to `doc_id`s.
- At query time, applies keyword pre-filtering to choose top candidate docs.
- Uses PageIndex `chat_completions` with citations enabled.
- Converts returned citations back into passage-level evidence objects for evaluation.

## 4) Data and artifact flow

Each pipeline has its own preprocessing and evaluation loop:

1. Read local HotpotQA dataset cache.
2. Build corpus and retrieval artifacts.
3. Run `run_evaluation.py` on sampled validation questions.
4. Save `evaluation_report.json`.

Pipeline-specific artifacts:

- `Dense_RAG`: `data/corpus.json`, `storage/`, `faiss_index/index.faiss`
- `Graph_RAG`: dense artifacts + `graph_index/graph.pkl` (+ `context_clusters.json`)
- `PageIndex_RAG`: `data/passage_lookup.json`, `pageindex_docs/`, `pageindex_index/doc_ids.json`

## 5) Project goals (practical + research)

- Establish a **strong dense retrieval baseline** for multi-hop QA.
- Test whether **graph expansion improves multi-hop evidence assembly**.
- Compare local-index pipelines against a **managed API approach** (PageIndex).
- Measure trade-offs in:
  - answer correctness,
  - support retrieval quality,
  - latency,
  - build complexity and infra cost.

## 6) What is already done vs what remains

### Already done
- Three runnable pipeline implementations exist.
- Preprocess/evaluate entry points are implemented in each variant.
- Metric computation is standardized across variants with minor graph additions.
- Artifacts and configuration are modularized per pipeline.

### Remaining / next milestones
- Add a **local LLM backend option** (Ollama) for Dense/Graph reader stage.
- Replace hardcoded provider secrets with environment variables.
- Add one top-level orchestrator script (optional) to benchmark all variants consistently.
- Add reproducibility notes (hardware profile, dataset snapshot, random seeds).

## 7) Local LLM (Ollama) fit for your plan

Your plan ("set up a local LLM and run this pipeline") is a good fit, especially for:

- `Dense_RAG` (recommended first),
- then `Graph_RAG`.

`PageIndex_RAG` is cloud-dependent by design and does not become fully local with Ollama.

### Minimal adaptation points for Ollama

For `Dense_RAG` and `Graph_RAG`, only the **reader** needs swapping:

- Current reader path: Groq client call in `reader.py`.
- Target reader path: local HTTP call to Ollama (`/api/generate` or `/api/chat`).
- Keep retrieval and evaluation untouched.

Suggested migration steps:

1. Add config fields:
   - `LLM_BACKEND = "groq" | "ollama"`
   - `OLLAMA_MODEL` (e.g., `llama3.1:8b`)
   - `OLLAMA_BASE_URL` (e.g., `http://localhost:11434`)
2. Refactor `generate_answer(...)` to route by backend.
3. Keep prompt format unchanged for fair comparison.
4. Re-run evaluation with same sample size and seed.

## 8) Operational runbook (recommended order)

1. Start with `Dense_RAG`.
2. Run `preprocess.py` once to build index artifacts.
3. Validate end-to-end on a small sample (`--samples 50` or `100`).
4. Integrate Ollama reader backend.
5. Re-run Dense evaluation and compare against Groq baseline.
6. Repeat for `Graph_RAG`.
7. Use `PageIndex_RAG` only when you specifically want managed-cloud comparison.

## 9) Risks / constraints to be aware of

- CPU-only preprocessing can be slow on fullwiki scale.
- Graph building increases memory/time footprint significantly.
- Cloud API variants introduce latency variability and external dependency.
- Current config files include inline API keys; treat as sensitive and rotate/move to env vars.

## 10) Success criteria for your next phase

You can consider the "local LLM phase" successful when:

- Dense/Graph pipelines run with Ollama without changing retrieval logic.
- `evaluation_report.json` is generated for both Groq and Ollama runs.
- You can compare EM/F1/Recall/latency side by side for backend choice.

---

If needed, the next implementation task is straightforward: add a backend-agnostic reader layer so Groq and Ollama can be switched by config in both `Dense_RAG` and `Graph_RAG`.
