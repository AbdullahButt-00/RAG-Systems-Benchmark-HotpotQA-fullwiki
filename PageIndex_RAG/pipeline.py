"""
End-to-end PageIndex-RAG pipeline: question → PageIndex query → answer + citations.
"""

import time

from pageindex_retriever import PageIndexRetriever


class PageIndexRAGPipeline:
    """
    Loads PageIndexRetriever on construction (builds keyword index from
    passage_lookup.json), then answers questions via the PageIndex API.

    Usage
    -----
    pipeline = PageIndexRAGPipeline()
    result = pipeline.run("Who directed Inception?")
    # result keys: answer, cited_passages, latency_ms, raw_responses
    """

    def __init__(self) -> None:
        self.retriever = PageIndexRetriever()

    def run(self, question: str) -> dict:
        """
        Run the full pipeline for a single question.

        Returns
        -------
        dict with:
            answer         (str)  — PageIndex-generated answer
            cited_passages (list) — passages cited by PageIndex
                                    each: {passage_id, title, text, citation_index}
            latency_ms     (float)— wall-clock time for pre-filter + API calls + parsing
            raw_responses  (list) — raw PageIndex API responses
        """
        t0 = time.perf_counter()
        result = self.retriever.query(question)
        latency_ms = (time.perf_counter() - t0) * 1_000

        return {
            "answer": result["answer"],
            "cited_passages": result["cited_passages"],
            "latency_ms": latency_ms,
            "raw_responses": result["raw_responses"],
        }
