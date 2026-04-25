"""
End-to-end Dense-RAG pipeline: question → retrieve → read → answer.
"""

import time

from config import TOP_K
from retriever import DenseRetriever
from reader import generate_answer


class DenseRAGPipeline:
    """
    Loads the persisted FAISS index on construction, then answers questions
    by dense retrieval followed by a Groq LLM reader.

    Usage
    -----
    pipeline = DenseRAGPipeline()
    result = pipeline.run("Who directed Inception?")
    # result keys: answer (str), retrieved_passages (list), latency_ms (float)
    """

    def __init__(self) -> None:
        self.retriever = DenseRetriever()
        self.retriever.load_index()

    def run(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Run the full pipeline for a single question.

        Returns
        -------
        dict with:
            answer (str)               — LLM-generated answer
            retrieved_passages (list)  — top_k passage dicts (passage_id, title, text, score)
            latency_ms (float)         — wall-clock time for retrieve + generate
        """
        t0 = time.perf_counter()
        passages = self.retriever.retrieve(question, top_k=top_k)
        answer = generate_answer(question, passages)
        latency_ms = (time.perf_counter() - t0) * 1_000

        return {
            "answer": answer,
            "retrieved_passages": passages,
            "latency_ms": latency_ms,
        }
