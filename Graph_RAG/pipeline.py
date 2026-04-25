"""
End-to-end Graph-RAG pipeline: question → seed → graph expand → read → answer.
"""

import json
import time

from config import TOP_K_SEED, TOP_K_GRAPH, MAX_HOP, CORPUS_PATH
from retriever import DenseRetriever
from graph_builder import load_graph
from graph_retriever import GraphRetriever
from reader import generate_answer


class GraphRAGPipeline:
    """
    Loads FAISS index and passage graph on construction, then answers questions
    via graph-guided multi-hop retrieval followed by a Groq LLM reader.

    Usage
    -----
    pipeline = GraphRAGPipeline()
    result = pipeline.run("Who was the first president of the United States?")
    # result keys: answer, retrieved_passages, seed_passages,
    #              expanded_passages, latency_ms
    """

    def __init__(self) -> None:
        print("Loading FAISS index...")
        dense = DenseRetriever()
        dense.load_index()

        print("Loading passage graph...")
        graph = load_graph()

        print("Loading corpus lookup...")
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            corpus: dict = json.load(f)

        self._retriever = GraphRetriever(dense, graph, corpus)

    def run(
        self,
        question: str,
        top_k_seed: int = TOP_K_SEED,
        top_k_graph: int = TOP_K_GRAPH,
        max_hop: int = MAX_HOP,
    ) -> dict:
        """
        Run the full pipeline for a single question.

        Returns
        -------
        dict with:
            answer             (str)        — LLM-generated answer
            retrieved_passages (list[dict]) — all passages ranked by score
            seed_passages      (list[dict]) — only the dense-seed passages
            expanded_passages  (list[dict]) — only the graph-expanded passages
            latency_ms         (float)      — wall-clock time for retrieve + generate
        """
        t0 = time.perf_counter()

        passages = self._retriever.retrieve(
            question,
            top_k_seed=top_k_seed,
            top_k_graph=top_k_graph,
            max_hop=max_hop,
        )
        answer = generate_answer(question, passages)

        latency_ms = (time.perf_counter() - t0) * 1_000

        return {
            "answer": answer,
            "retrieved_passages": passages,
            "seed_passages": [p for p in passages if p["is_seed"]],
            "expanded_passages": [p for p in passages if not p["is_seed"]],
            "latency_ms": latency_ms,
        }
