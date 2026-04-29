# RAG Systems Benchmark Report (HotpotQA FullWiki)

---

## 1. Overview

This repository benchmarks four Retrieval-Augmented Generation (RAG) pipeline variants on the **HotpotQA fullwiki** dataset — a multi-hop question answering benchmark requiring evidence from two or more Wikipedia passages to answer correctly. All pipelines share the same deduplicated corpus (~508k passages), evaluation harness (500 validation samples), and core metrics (EM, Token F1, SP-F1, Recall@k, MRR, latency), enabling a controlled comparison across retrieval architectures.

### Approaches Summarized

| Folder | Approach | Retrieval | Reader | Status |
|---|---|---|---|---|
| `Dense_RAG` | Dense FAISS baseline | FAISS flat L2 top-5 | Ollama (local) | Evaluated |
| `Graph_RAG` | Graph-guided multi-hop | FAISS seeds + BFS expansion | Groq llama-3.3-70b | Evaluated |
| `Hybrid_RAG` | Confidence-gated staged hybrid | FAISS + conditional BFS + reranker | Groq llama-3.3-70b | Evaluated |
| `PageIndex_RAG` | Cloud API-driven | Keyword pre-filter + PageIndex API | PageIndex built-in | Not evaluated |

---

## 2. Approaches Breakdown

---

### 2.1 Dense_RAG

#### Methodology and System Architecture

Dense_RAG is the baseline pipeline. It encodes all corpus passages into dense 384-dimensional vectors using a sentence-transformer model and indexes them in a FAISS flat L2 index. At query time, it encodes the question, performs an approximate nearest-neighbor search over the index, retrieves the top-5 most similar passages, and passes them as context to a local LLM reader (Ollama) to generate a short factual answer.

**Step-by-step pipeline:**

```
Question
  │
  ▼
[HuggingFaceEmbedding: multi-qa-MiniLM-L6-cos-v1]  →  384-dim query vector
  │
  ▼
[FAISS IndexFlatL2] → top-5 passage_ids + L2 similarity scores
  │
  ▼
[Corpus lookup: corpus.json] → {passage_id, title, text, score}
  │
  ▼
[Ollama reader: llama3] → short factual answer string
```

#### Dataset and Preprocessing

- **Source:** HotpotQA fullwiki, loaded directly from local Arrow files (HuggingFace cache format)
- **Splits used:** Train + Validation (both used for corpus construction; only Validation for evaluation)
- **Corpus size:** ~508k deduplicated passages, deduplicated by title (first occurrence wins)
- **Passage format:** `title + ": " + joined sentences` — title prepended to passage text
- **Chunking:** One LlamaIndex `Document` per passage (no sub-document chunking); `SentenceSplitter(chunk_size=1024, chunk_overlap=0)` configured but passages are already shorter than 1024 tokens
- **Indexing:** `FAISS IndexFlatL2(384)` via `llama-index-vector-stores-faiss`; docstore and index metadata persisted to `storage/`; raw FAISS binary separately to `faiss_index/index.faiss`
- **Evaluation set:** 500 randomly sampled validation questions (default)

#### Tools and Technologies Used

| Component | Tool / Library |
|---|---|
| Embedding model | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (384-dim) |
| Embedding framework | `llama-index-embeddings-huggingface` |
| Vector index | `FAISS IndexFlatL2` via `llama-index-vector-stores-faiss` |
| Index abstraction | `LlamaIndex VectorStoreIndex` |
| Dataset loading | HuggingFace `datasets` (Arrow format) |
| LLM reader | Ollama local API (`/api/generate`) |
| Prompt format | System instruction + numbered passage context + "Question: ... Answer:" |
| Numeric compute | NumPy |

#### Experimental Setup and Metrics

- **Evaluation samples:** 500 validation questions
- **Top-k retrieved:** 5 passages per question
- **Reader temperature:** 0.0 (deterministic)
- **Metrics computed:** EM, Token F1, Supporting Fact F1, Recall@5, MRR, Mean Latency (ms), P95 Latency (ms)
- **EM normalization:** lowercase, remove articles (a/an/the), strip punctuation, collapse whitespace
- **SP-F1:** Title-level precision/recall/F1 between retrieved passage titles and gold supporting fact titles
- **MRR:** 1/rank of the first retrieved passage whose title matches any gold supporting fact title

#### Results and Discussion

| Metric | Value |
|---|---|
| Exact Match (EM) | **0.084** (8.4%) |
| Token F1 | **0.1452** (14.52%) |
| Supporting Fact F1 | **0.0509** (5.09%) |
| Recall@5 | **0.162** (16.2%) |
| MRR | **0.1463** (14.63%) |
| Mean Latency | **48,206 ms** (48.2 sec/question) |
| P95 Latency | **80,049 ms** (80.0 sec/question) |

**Discussion:** Dense_RAG achieves the highest MRR (0.1463) of all three evaluated pipelines, indicating that when it retrieves a relevant passage at all, that passage tends to appear near rank 1. However, low Recall@5 (16.2%) reveals that for the majority of HotpotQA multi-hop questions, none of the 5 retrieved passages contain a gold supporting fact title. This is expected: single-hop dense retrieval is structurally limited for questions requiring evidence from two non-directly-related passages. The very high latency (~48 sec/question mean) is driven by the local Ollama reader rather than retrieval, reflecting CPU-bound LLM inference.

---

### 2.2 Graph_RAG

#### Methodology and System Architecture

Graph_RAG extends Dense_RAG by adding a passage-level knowledge graph. After FAISS retrieval produces 5 seed passages, a Breadth-First Search (BFS) over a NetworkX directed graph expands the evidence pool up to 2 hops away, returning up to 10 total passages. The intuition is that multi-hop questions require following entity or title links across passages — a structure dense retrieval cannot capture in a single step.

**Step-by-step pipeline:**

```
Question
  │
  ▼
[FAISS top-5 seed retrieval]  →  5 seed passage_ids + scores
  │
  ▼
[NetworkX BFS expansion]
  │  max_hop=2, start from seed nodes, follow successors
  │  edge_score = edge_weight / (hop_distance + 2)
  │
  ▼
[Score-ranked merge]  →  top-10 passages (seeds union expanded)
  │
  ▼
[Corpus text attachment]  →  {passage_id, title, text, sentences, score, hop_distance, is_seed}
  │
  ▼
[Groq llama-3.3-70b-versatile reader]  →  answer string
```

**Graph construction (three edge types):**

| Edge Type | Weight | Construction |
|---|---|---|
| `title_link` | 1.0 | Directed: P→Q if passage Q's title appears verbatim in passage P's text |
| `entity_overlap` | min(1.0, shared_count/5) | Undirected: passages sharing ≥1 capitalised multi-word phrase |
| `same_context` | 0.5 | Undirected: passages co-appearing in the same HotpotQA question's context list |

Entity edges cap at `_MAX_PASSAGES_PER_ENTITY = 50` to prevent high-frequency entities (e.g., "United States") from generating O(n²) pairs. BFS node scores decay as `weight / (hop + 2)`.

#### Dataset and Preprocessing

- **Source:** HotpotQA fullwiki (same Arrow files as Dense_RAG)
- **Corpus:** ~508k deduplicated passages — identical to Dense_RAG corpus
- **Additional artifact:** `data/context_clusters.json` — per-question lists of passage_ids that co-appeared in HotpotQA context fields; used to build `same_context` edges
- **Graph persistence:** NetworkX DiGraph serialized via `pickle.HIGHEST_PROTOCOL` to `graph_index/graph.pkl`
- **Build time:** ~15 min for FAISS indexing + 15–45 min for graph construction (~30–60 min total)

#### Tools and Technologies Used

| Component | Tool / Library |
|---|---|
| Dense index | FAISS + LlamaIndex (identical to Dense_RAG) |
| Graph engine | NetworkX `DiGraph` |
| Graph persistence | Python `pickle` |
| Entity extraction | Regex: `\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b` |
| LLM reader | Groq API (`llama-3.3-70b-versatile`) |
| Additional metric | `chain_recall_at_10`, `support_coverage` |

#### Experimental Setup and Metrics

- **Seeds:** top_k_seed=5 (FAISS)
- **Expanded:** top_k_graph=10 (total after BFS)
- **BFS depth:** max_hop=2
- **Reader:** Groq `llama-3.3-70b-versatile`, temperature not specified (default)
- **Extra metrics:** Chain Recall@10 (all gold titles present in top-10), Support Coverage (sentence-level coverage of gold supporting facts)

#### Results and Discussion

| Metric | Value |
|---|---|
| Exact Match (EM) | **0.070** (7.0%) |
| Token F1 | **0.1256** (12.56%) |
| Supporting Fact F1 | **0.0320** (3.2%) |
| Recall@5 | **0.152** (15.2%) |
| MRR | **0.0403** (4.03%) |
| Chain Recall@10 | **0.026** (2.6%) |
| Support Coverage | **0.0988** (9.88%) |
| Mean Latency | **89,676 ms** (89.7 sec/question) |
| P95 Latency | **130,917 ms** (130.9 sec/question) |

**Discussion:** Graph_RAG performs **worse than Dense_RAG on every primary metric**. EM drops from 8.4% to 7.0%, Token F1 from 14.52% to 12.56%, and MRR collapses from 0.1463 to 0.0403. This counter-intuitive result has several causes. First, BFS expansion from 5 seeds into a graph of 508k nodes introduces significant noise — many entity-overlap and same-context neighbors are topically unrelated to the question. Second, the hop-decay scoring formula (`weight / (hop + 2)`) systematically assigns lower scores to expanded nodes, but the ranking still places many irrelevant expanded passages ahead of the seeds that were actually relevant, hurting MRR. Third, the `same_context` edges introduce dataset leakage risk (passages grouped by dataset construction, not semantic relevance). Chain Recall@10 of 2.6% confirms that BFS rarely assembles the complete multi-hop evidence chain. Mean latency nearly doubles Dense_RAG at 89.7 sec, driven by the cost of BFS traversal over a large graph on top of Groq API response time.

---

### 2.3 Hybrid_RAG

#### Methodology and System Architecture

Hybrid_RAG is the most architecturally sophisticated pipeline. It introduces a **confidence gate** that selectively triggers graph expansion only when the dense retriever's score distribution signals uncertainty. This avoids Graph_RAG's blanket expansion noise while preserving the ability to assemble multi-hop evidence for ambiguous queries. It also adds an **evidence reranker** that normalizes and fuses dense and graph scores before passing the final 7 passages to the LLM.

The graph used is **leakage-free**: it excludes `same_context` edges present in Graph_RAG's graph, using only `title_link` and `entity_overlap` edges derived purely from passage text.

**Step-by-step pipeline:**

```
Question
  │
  ▼
Stage 1: DenseRetriever.retrieve_seed(top_k=5)
  │       FAISS IndexFlatL2, multi-qa-MiniLM-L6-cos-v1 (384-dim)
  │
  ▼
Stage 2: QueryAnalyzer.analyze(seeds)
  │       confidence = max(seed_scores)
  │       spread     = max(seed_scores) - min(seed_scores)
  │       gate_fired = (confidence < 0.40) OR (spread < 0.05)
  │
  ├── gate_fired=False ──────────────────────────────────────┐
  │                                                          │
  ▼ gate_fired=True                                          │
Stage 3: GraphRetriever.expand_from_seeds(seeds, top_k=10, hops=2)
  │       BFS over title_link + entity_overlap edges only    │
  │       (same_context edges excluded)                      │
  │                                                          │
  ▼                                                          ▼
Stage 4: EvidenceReranker.rerank(seeds, expanded, top_k=7)
  │       dense_norm = minmax_normalize(seed_scores)
  │       graph_norm = minmax_normalize(expanded_scores)
  │       final_score = 0.6 * dense_norm  (seeds)
  │                   = 0.4 * graph_norm  (expanded)
  │
  ▼
Stage 5: reader.generate_answer(question, top_7_passages)
  │       Groq llama-3.3-70b-versatile, temp=0.0, max_tokens=128
  │
  ▼
Answer
```

**Gate logic detail:** LlamaIndex's FaissVectorStore returns scores as negative L2 distances (range: `[-1.41, 0]`). With `CONF_THRESHOLD=0.40` and all scores being negative, the confidence condition (`max_score < 0.40`) always evaluates true. The gate therefore fires exclusively when `spread < SPREAD_THRESHOLD=0.05` — confirmed by the evaluation: all 54 gate firings were classified as `low_spread`, never `low_confidence`.

#### Dataset and Preprocessing

- **Source:** HotpotQA fullwiki (same corpus as Dense_RAG and Graph_RAG)
- **Graph:** Leakage-free DiGraph — rebuilt from scratch, identical edge types to Graph_RAG minus `same_context`
- **Build options:** Full rebuild (~75 min) or reuse Dense_RAG/Graph_RAG FAISS artifacts and rebuild graph only
- **Evaluation set:** 500 validation questions

#### Tools and Technologies Used

| Component | Tool / Library |
|---|---|
| Dense index | FAISS + LlamaIndex (identical to Dense_RAG) |
| Graph engine | NetworkX `DiGraph` (no same_context edges) |
| Confidence gate | Custom `QueryAnalyzer` (score statistics) |
| Reranker | Custom `EvidenceReranker` (min-max normalization + weighted sum) |
| LLM reader | Groq API (`llama-3.3-70b-versatile`, temp=0.0, max_tokens=128) |
| Ablation support | CLI flags: `--conf_threshold`, `--spread_threshold`, `--alpha` |
| Demo | `demo.py` interactive single-question runner with verbose diagnostics |

#### Experimental Setup and Metrics

- **Config:** `conf_threshold=0.40`, `spread_threshold=0.05`, `rerank_alpha=0.6`
- **Top-k seed:** 5, **Top-k graph:** 10, **Top-k final:** 7
- **Gate behavior:** 54/500 queries triggered expansion (10.8%), all due to `low_spread`
- **Average expanded passages when gate fired:** 9.81
- **Metrics (9 total):** EM, Token F1, SP-F1, Recall@5, Recall@7, MRR, Chain Recall, Support Coverage, Gate Firing Rate

#### Results and Discussion

| Metric | Value |
|---|---|
| Exact Match (EM) | **0.106** (10.6%) |
| Token F1 | **0.1785** (17.85%) |
| Supporting Fact F1 | **0.0507** (5.07%) |
| Recall@5 | **0.162** (16.2%) |
| Recall@7 | **0.162** (16.2%) |
| MRR | **0.0403** (4.03%) |
| Chain Recall | **0.016** (1.6%) |
| Support Coverage | **0.000** (0.0%) |
| Mean Latency | **3,925 ms** (3.9 sec/question) |
| P95 Latency | **7,224 ms** (7.2 sec/question) |
| Gate Firing Rate | **0.108** (10.8%) |

**Discussion:** Hybrid_RAG achieves the best EM (10.6%) and Token F1 (17.85%) across all evaluated pipelines, while also being the fastest — 12× faster than Dense_RAG and 23× faster than Graph_RAG. The speed advantage stems from using Groq (cloud API) rather than local Ollama, and from graph expansion firing only 10.8% of the time (89.2% of queries are handled by dense-only retrieval + Groq, a fast path).

MRR (0.0403) is identical to Graph_RAG's and significantly lower than Dense_RAG's (0.1463). This indicates that the reranker, despite improving answer quality (EM/F1), pushes gold passages down in the ranked list — the reranked ordering is not optimal for retrieval-level metrics.

**Support Coverage = 0.0** is anomalous and likely reflects a metric computation bug: the hybrid pipeline attaches `sentences` to seed passages from corpus lookup, but expanded passages may lack this field, causing sentence-level matching to fail for the entire result set. This is a measurement artifact rather than a true coverage result of zero.

The gate miscalibration (CONF_THRESHOLD=0.40 vs negative L2 scores) means the confidence trigger never fires — the gate is effectively a spread-only gate. This is acknowledged in `query_analyzer.py` with an explicit warning and a suggested fix. A properly calibrated negative confidence threshold would change the gate firing rate and potentially affect results.

---

### 2.4 PageIndex_RAG

#### Methodology and System Architecture

PageIndex_RAG replaces the entire local retrieval and generation stack with a cloud-hosted API. Passages are pre-processed into plain-text bundle files (50 passages each, ~10,160 bundles total), uploaded to the PageIndex service, and assigned `doc_id`s. At query time, a keyword-based inverted index (built in-memory at startup from passage titles) pre-selects up to 10 candidate documents. Only those candidate documents are queried via `PageIndex chat_completions` with `enable_citations=True`. The response with the most citations is selected as the answer.

**Step-by-step pipeline:**

```
[Startup]
  passage_lookup.json → tokenize titles → keyword → [doc_id] inverted index

[Query time]
Question
  │
  ▼
Tokenize question (lowercase, strip punct, remove stopwords)
  │
  ▼
Keyword lookup → score doc_ids by hit count → top-10 candidate doc_ids
  │
  ▼
For each candidate doc_id:
  PageIndex.chat_completions(question, doc_id, enable_citations=True, temp=0)
  │
  ▼
Merge: select response with most citations
  │
  ▼
Parse citations → {passage_id, title, text} via:
  1. [PASSAGE_ID: ...] marker regex
  2. [TITLE: ...] marker regex
  3. Fallback: substring word-overlap against passage text
  │
  ▼
Answer + cited_passages
```

#### Dataset and Preprocessing

- **Source:** HotpotQA fullwiki (~508k passages)
- **Bundle format:** Each bundle is a plain-text file with `PASSAGES_PER_DOCUMENT=50` passages, each marked with `[PASSAGE_ID: ...]` and `[TITLE: ...]` structured markers
- **Number of bundles:** ~10,160 uploaded documents
- **Lookup artifact:** `data/passage_lookup.json` — maps `passage_id` → `{title, text, bundle_file, doc_id}`
- **Index artifact:** `pageindex_index/doc_ids.json` — maps bundle filename → PageIndex `doc_id`
- **Pre-filter reduction:** From ~10,160 possible documents to ≤10 per query via keyword scoring
- **Build time:** ~5–10 min (corpus + bundles, no embedding), ~30 min for uploads (PageIndex API throughput)

#### Tools and Technologies Used

| Component | Tool / Library |
|---|---|
| Cloud API | PageIndex (`pageindex` Python client) |
| Inverted index | Custom keyword index (in-memory dict at startup) |
| Stopword filter | Hardcoded 40-word frozenset |
| Citation parsing | Regex + fallback substring matching |
| Dataset loading | HuggingFace `datasets` |
| Dependencies | Minimal: `pageindex`, `datasets`, `numpy`, `tqdm` |
| Local footprint | ~100 MB (bundle files only, no FAISS/torch/networkx) |

#### Experimental Setup and Metrics

- **Evaluation status:** **No `evaluation_report.json` exists.** The evaluation has not been run.
- **Planned metrics:** EM, Token F1, SP-F1, Recall@5, MRR, Mean Latency, P95 Latency (identical logic to Dense_RAG `evaluate.py`, copied verbatim)
- **Planned samples:** 500 (default)
- **Key constraint:** Evaluation requires an active PageIndex API key and uploaded bundles; the `pageindex_index/doc_ids.json` file exists, indicating bundles were uploaded, but evaluation was not executed.

---

## 3. Comparative Analysis

### 3.1 Metrics Comparison Table

| Metric | Dense_RAG | Graph_RAG | Hybrid_RAG | PageIndex_RAG |
|---|---|---|---|---|
| Exact Match (EM) | 0.084 | 0.070 | **0.106** | N/A |
| Token F1 | 0.1452 | 0.1256 | **0.1785** | N/A |
| Supporting Fact F1 | 0.0509 | 0.0320 | 0.0507 | N/A |
| Recall@5 | 0.162 | 0.152 | 0.162 | N/A |
| MRR | **0.1463** | 0.0403 | 0.0403 | N/A |
| Chain Recall@10 | — | 0.026 | 0.016 | N/A |
| Support Coverage | — | 0.0988 | 0.000\* | N/A |
| Mean Latency (ms) | 48,206 | 89,676 | **3,925** | N/A |
| P95 Latency (ms) | 80,049 | 130,917 | **7,224** | N/A |
| Gate Firing Rate | — | — | 0.108 | — |
| Passages to LLM | 5 | 10 | 7 | varies |
| LLM Backend | Ollama (local) | Groq API | Groq API | PageIndex API |

\* *Anomalous — likely a metric computation artifact; see Section 2.3 discussion.*

### 3.2 Key Insights and Trade-offs

**EM and Token F1 — Hybrid_RAG wins, but by a moderate margin.**
Hybrid_RAG's EM (10.6%) is only 2.2 points above Dense_RAG (8.4%) and 3.6 points above Graph_RAG (7.0%). Absolute performance across all three is low, reflecting the fundamental difficulty of HotpotQA fullwiki multi-hop questions for RAG systems without specialized multi-hop reasoning chains.

**MRR — Dense_RAG wins, exposing a reranker flaw.**
Dense_RAG's MRR (0.1463) is 3.6× higher than both Graph_RAG and Hybrid_RAG (0.0403). This means FAISS top-1 is much more likely to be a gold passage than after graph expansion or reranking. Graph BFS expansion and min-max reranking push gold passages down the ranked list even when they were retrieved correctly at position 1 or 2 as seeds. This is a significant finding: both graph-based pipelines improve answer quality (EM) while simultaneously degrading retrieval precision (MRR), suggesting the LLM reader compensates for a noisier retrieved context through its own reasoning.

**Latency — Hybrid_RAG dominates, but the comparison is confounded.**
The primary latency driver is the **reader backend**, not the retrieval complexity:
- Dense_RAG uses local Ollama (CPU inference) → ~48 sec/question
- Graph_RAG uses Groq API + heavy BFS → ~89 sec/question (BFS overhead + API latency)
- Hybrid_RAG uses Groq API + sparse BFS (10.8% gate rate) → ~3.9 sec/question

Hybrid_RAG's latency advantage is partly architectural (confidence gate avoids expansion 89.2% of the time) and partly due to using a fast cloud API vs local CPU LLM. A fair comparison would require a consistent reader backend across all three.

**Graph expansion is not helping retrieval metrics.**
Across both Graph_RAG and Hybrid_RAG, SP-F1 (~0.032–0.051) and Recall@5 (0.152–0.162) barely differ from Dense_RAG. Chain Recall@10 (2.6% in Graph_RAG, 1.6% in Hybrid_RAG) confirms that BFS rarely assembles the complete two-passage evidence chain required by HotpotQA questions. The graph edges are mostly noisy: entity-overlap edges link passages sharing common named entities regardless of question relevance, and the hop-decay scoring does not adequately suppress irrelevant neighbors.

**Gate miscalibration is a critical issue in Hybrid_RAG.**
The confidence gate fires only on `low_spread` (10.8% of queries) because `CONF_THRESHOLD=0.40` is incompatible with the negative L2 distance scores returned by LlamaIndex's FAISS backend (actual range: ~`[-1.41, 0]`). The confidence branch of the gate is non-functional in this run. Despite this, EM improves — indicating that the spread-based trigger alone selects genuinely ambiguous queries where expansion helps.

---

## 4. Key Findings

**1. Dense FAISS retrieval is the most precise single-stage retriever (MRR=0.1463).**
When relevant passages exist in the top-5, dense retrieval ranks them at position 1 more reliably than any graph-augmented pipeline. Graph expansion adds noise that degrades rank precision.

**2. Graph expansion hurts retrieval metrics but can improve answer quality.**
Graph_RAG degrades on all metrics vs Dense_RAG. Hybrid_RAG partially recovers answer quality (EM, Token F1) through selective gating and reranking, but retrieval metrics (MRR, SP-F1) remain degraded relative to the Dense baseline.

**3. The fundamental bottleneck is Recall@5 / Recall@7 (≤16.2%).**
Across all pipelines, fewer than 1 in 5 questions have any gold supporting fact in the retrieved set. This ceiling directly limits EM and Token F1. No architectural layer (graph, gate, reranker) compensates for retrieval failing to include the evidence at all.

**4. Entity-overlap edges are too noisy for multi-hop evidence assembly.**
Chain Recall@10 of 2.6% (Graph_RAG) and 1.6% (Hybrid_RAG) shows that BFS over entity-overlap edges does not reliably connect the two Wikipedia articles required per HotpotQA question. Title-link edges are semantically precise but sparse; entity-overlap edges are dense but imprecise.

**5. The same_context edges in Graph_RAG introduce dataset leakage.**
Edges built from co-occurring passages in HotpotQA context fields encode ground-truth question structure. Hybrid_RAG correctly excludes these, resulting in a cleaner evaluation, though the impact on metric inflation was not directly ablated in the evaluated runs.

**6. LLM reader quality is confounded with retrieval in the latency and EM comparisons.**
Dense_RAG (Ollama) vs Graph/Hybrid_RAG (Groq llama-3.3-70b) are not using equivalent readers. The larger Groq model (70B) likely contributes to Hybrid_RAG's higher EM, independent of the retrieval architecture.

**7. PageIndex_RAG remains unvalidated.**
The pipeline is fully implemented and bundles are uploaded, but no evaluation has been run. Its cloud-API design trades local infrastructure cost for external dependency and latency variability.

---

## 5. Recommendations / Future Improvements

**R1 — Fix the confidence gate threshold calibration (Hybrid_RAG).**
Run `query_analyzer.run_threshold_diagnostic()` on 100 validation examples to inspect the actual score distribution (expected range: `[-1.41, 0]`). Set `CONF_THRESHOLD` to a negative value (e.g., `-0.70`) that separates confident from uncertain retrievals. This will activate the confidence branch of the gate and may change the gate firing rate and downstream metrics substantially.

**R2 — Standardize the LLM reader across all pipelines.**
Replace Dense_RAG's Ollama reader with Groq `llama-3.3-70b-versatile` (or vice versa) to isolate retrieval quality from reader quality in the comparison. Currently EM differences partly reflect a 70B vs smaller model gap, not retrieval architecture.

**R3 — Replace entity-overlap edges with a co-reference or entity-linking method.**
The regex-based `[A-Z][a-z]+` entity extractor is noisy and creates edges between passages sharing common proper nouns unrelated to the question. Replacing it with a named entity linker (e.g., spaCy + WikiData) would improve edge precision and likely improve Chain Recall.

**R4 — Run and evaluate PageIndex_RAG.**
Execute `python run_evaluation.py` in `PageIndex_RAG/` with a valid API key to get comparable metrics. This is the only pipeline with a cloud-native retrieval+generation architecture and its SP-F1 and latency would provide a meaningful comparison point against local approaches.

**R5 — Add a multi-hop iterative retriever as a stronger baseline.**
Current pipelines retrieve in a single step (Dense) or expand blindly (Graph). An iterative retriever — which uses the LLM to generate a sub-question after the first retrieval step, then retrieves again — would be far more effective for HotpotQA bridge questions. This is the approach used by state-of-the-art systems like IRCoT and Self-Ask.

**R6 — Increase evaluation sample size to 7405 (full validation set).**
500-sample results have high variance on low-frequency metrics like Chain Recall (2.6% = 13 questions). Full validation evaluation would reduce variance and produce more reliable cross-pipeline comparisons.

**R7 — Add a dedicated ablation for same_context edge leakage.**
Run Graph_RAG's evaluation using Hybrid_RAG's leakage-free graph and compare SP-F1 and Chain Recall. This would directly quantify how much of Graph_RAG's Support Coverage (9.88%) is inflated by dataset-structure edges.

**R8 — Replace inline API keys with environment variables.**
`GROQ_API_KEY` and `PAGEINDEX_API_KEY` are currently hardcoded in `config.py` files. Move to `os.environ` reads with a `.env` file to prevent accidental credential exposure in version control.
