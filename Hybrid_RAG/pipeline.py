"""
Confidence-Gated Staged Hybrid RAG — end-to-end pipeline.

Pipeline stages
---------------
  Stage 1  Dense seed retrieval    DenseRetriever.retrieve_seed()
  Stage 2  Confidence gate         QueryAnalyzer.analyze()
  Stage 3  [conditional] BFS       GraphRetriever.expand_from_seeds()
  Stage 4  Evidence reranking      EvidenceReranker.rerank()
  Stage 5  LLM generation          reader.generate_answer()

The gate (Stage 2) is the distinguishing component. When the dense
score distribution signals high confidence, Stage 3 is skipped and the
pipeline behaves like Dense_RAG with a unified reranker. When the gate
fires, graph expansion augments the seed pool before reranking.

Usage
-----
    pipeline = HybridRAGPipeline()
    result = pipeline.run("Who directed the film that stars Tom Hanks as Forrest Gump?")
    print(result["answer"])
    print(result["gate_analysis"])   # {'gate_fired': True, 'confidence': 0.31, ...}
"""

import json
import time

from config import (
    CONF_THRESHOLD,
    CORPUS_PATH,
    MAX_HOP,
    RERANK_ALPHA,
    SPREAD_THRESHOLD,
    TOP_K_FINAL,
    TOP_K_GRAPH,
    TOP_K_SEED,
)
from retriever      import DenseRetriever
from graph_builder  import load_graph
from graph_retriever import GraphRetriever
from query_analyzer import QueryAnalyzer
from reranker       import EvidenceReranker
from reader         import generate_answer


class HybridRAGPipeline:
    """
    Confidence-Gated Staged Hybrid RAG pipeline.

    Loads FAISS index, passage graph, and corpus on construction.
    All subsequent calls to run() reuse these loaded objects.

    Parameters
    ----------
    conf_threshold : float
        Confidence gate threshold (overrides config). None = use config value.
    spread_threshold : float
        Spread gate threshold (overrides config). None = use config value.
    rerank_alpha : float
        Dense score weight in reranker (overrides config). None = use config value.
    """

    def __init__(
        self,
        conf_threshold:   float | None = None,
        spread_threshold: float | None = None,
        rerank_alpha:     float | None = None,
    ) -> None:
        # ── Load FAISS index ───────────────────────────────────────────────
        print("Loading FAISS index...")
        self._dense = DenseRetriever()
        self._dense.load_index()

        # ── Load leakage-free passage graph ───────────────────────────────
        print("Loading hybrid passage graph (title_link + entity_overlap only)...")
        graph = load_graph()

        # ── Load corpus for text lookup and sentences attachment ──────────
        print("Loading corpus lookup...")
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            self._corpus: dict = json.load(f)

        # ── Initialise pipeline components ────────────────────────────────
        self._graph_retriever = GraphRetriever(self._dense, graph, self._corpus)

        self._analyzer = QueryAnalyzer(
            conf_threshold   = conf_threshold   if conf_threshold   is not None else CONF_THRESHOLD,
            spread_threshold = spread_threshold if spread_threshold is not None else SPREAD_THRESHOLD,
        )

        self._reranker = EvidenceReranker(
            alpha = rerank_alpha if rerank_alpha is not None else RERANK_ALPHA,
        )

        print("Pipeline ready.\n")

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def run(
        self,
        question:     str,
        top_k_seed:   int = TOP_K_SEED,
        top_k_graph:  int = TOP_K_GRAPH,
        top_k_final:  int = TOP_K_FINAL,
        max_hop:      int = MAX_HOP,
    ) -> dict:
        """
        Run the full hybrid pipeline for a single question.

        Parameters
        ----------
        question    : str  — the natural language question
        top_k_seed  : int  — number of dense seeds (Stage 1)
        top_k_graph : int  — max expanded passages from BFS (Stage 3)
        top_k_final : int  — passages passed to LLM reader (Stage 4→5)
        max_hop     : int  — BFS depth limit

        Returns
        -------
        dict with:
            answer              (str)        — LLM-generated answer
            retrieved_passages  (list[dict]) — final reranked passages (sent to LLM)
            seed_passages       (list[dict]) — raw dense seeds (before reranking)
            expanded_passages   (list[dict]) — graph-expanded passages ([] if gate didn't fire)
            gate_analysis       (dict)       — {gate_fired, confidence, spread, trigger_reason}
            n_seeds             (int)        — number of seeds retrieved
            n_expanded          (int)        — number of expanded passages
            n_final             (int)        — number of passages sent to LLM
            latency_ms          (float)      — wall-clock time (retrieve + generate)
        """
        t0 = time.perf_counter()

        # ── Stage 1: Dense seed retrieval ─────────────────────────────────
        seeds = self._dense.retrieve_seed(question, top_k=top_k_seed)

        # Attach sentences to seeds from corpus.json.
        # DenseRetriever does not include sentences (they're not in node
        # metadata to keep FAISS node size small). The support_coverage
        # metric requires sentences, so we fetch them here.
        for s in seeds:
            pid = s.get("passage_id", "")
            if pid and "sentences" not in s:
                entry = self._corpus.get(pid, {})
                s["sentences"] = entry.get("sentences", [])

        # ── Stage 2: Confidence gate ───────────────────────────────────────
        gate = self._analyzer.analyze(seeds)

        # ── Stage 3: Conditional graph expansion ──────────────────────────
        if gate["gate_fired"]:
            expanded = self._graph_retriever.expand_from_seeds(
                seeds,
                top_k_graph=top_k_graph,
                max_hop=max_hop,
            )
        else:
            expanded = []

        # ── Stage 4: Evidence reranking ────────────────────────────────────
        reranked = self._reranker.rerank(seeds, expanded, top_k=top_k_final)

        # ── Stage 5: LLM generation ────────────────────────────────────────
        answer = generate_answer(question, reranked)

        latency_ms = (time.perf_counter() - t0) * 1_000

        return {
            "answer":             answer,
            "retrieved_passages": reranked,
            "seed_passages":      seeds,
            "expanded_passages":  expanded,
            "gate_analysis":      gate,
            "n_seeds":            len(seeds),
            "n_expanded":         len(expanded),
            "n_final":            len(reranked),
            "latency_ms":         latency_ms,
        }
