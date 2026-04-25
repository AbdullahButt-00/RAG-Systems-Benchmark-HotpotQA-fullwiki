"""
Evidence reranker — merges dense seeds and graph-expanded passages into
a single ranked list for the LLM reader.

Scoring formula
---------------
Dense seeds and graph-expanded passages have scores on different scales:

  Dense seeds   : L2-derived similarity scores (typical range 0.1–0.9)
  Graph expanded: edge_weight / (hop+2)        (typical range 0.04–0.5)

Direct comparison of these raw values would systematically favor
seeds over expanded passages regardless of relevance.

Solution: min-max normalize each pool independently to [0, 1],
then combine with a weighted sum:

  For seed passage i:
      final_score = RERANK_ALPHA * dense_norm[i]

  For expanded passage j:
      final_score = (1 - RERANK_ALPHA) * graph_norm[j]

  All passages are then ranked by final_score descending.
  Top TOP_K_FINAL passages are returned to the LLM reader.

Ablation hooks
--------------
  alpha=1.0  → dense-only ordering (seeds rank above all expanded)
  alpha=0.0  → graph-only ordering (expanded rank above all seeds)
  alpha=0.6  → default: mild bias toward dense (recommended)

Edge case: no expanded passages
--------------------------------
If the gate did not fire (gate_fired=False), expanded=[] and the reranker
falls back to dense-only scoring. This is mathematically equivalent to
Dense_RAG's ranking but with normalized scores instead of raw scores.
"""

from config import RERANK_ALPHA, TOP_K_FINAL


def _minmax_normalize(scores: list[float]) -> list[float]:
    """
    Scale scores to [0, 1].

    If all scores are equal (denom ≈ 0), returns [1.0]*n so that
    all equally-scored passages are treated uniformly rather than zeroed out.
    """
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    denom = mx - mn
    if denom < 1e-9:
        return [1.0] * len(scores)
    return [(s - mn) / denom for s in scores]


class EvidenceReranker:
    """
    Merges dense seed passages and graph-expanded passages using
    independent score normalization and a weighted combination.

    Parameters
    ----------
    alpha : float
        Weight assigned to normalized dense scores (0.0–1.0).
        (1 - alpha) is assigned to normalized graph scores.
        Default: RERANK_ALPHA from config.py (0.6)
    """

    def __init__(self, alpha: float = RERANK_ALPHA) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def rerank(
        self,
        seeds: list[dict],
        expanded: list[dict],
        top_k: int = TOP_K_FINAL,
    ) -> list[dict]:
        """
        Produce a unified top-k passage list from seeds and expanded passages.

        Each returned passage has two additional fields added:
            final_score (float) — combined normalized score used for ranking
            source      (str)   — 'dense_seed' or 'graph_expanded'

        Parameters
        ----------
        seeds : list[dict]
            Dense seed passages with 'score' key (from DenseRetriever).
        expanded : list[dict]
            Graph-expanded passages with 'score' key (from GraphRetriever).
            May be empty (gate did not fire).
        top_k : int
            Number of passages to return.

        Returns
        -------
        list[dict] — sorted by final_score descending, length ≤ top_k.
        """
        if not seeds and not expanded:
            return []

        # ── No expansion: dense-only path ─────────────────────────────────
        if not expanded:
            dense_norms = _minmax_normalize([p["score"] for p in seeds])
            results = []
            for p, dn in zip(seeds, dense_norms):
                entry = dict(p)
                entry["final_score"] = self.alpha * dn
                entry["source"]      = "dense_seed"
                results.append(entry)
            results.sort(key=lambda x: x["final_score"], reverse=True)
            return results[:top_k]

        # ── Full hybrid path ───────────────────────────────────────────────
        dense_norms = _minmax_normalize([p["score"] for p in seeds])
        graph_norms = _minmax_normalize([p["score"] for p in expanded])

        results = []

        for p, dn in zip(seeds, dense_norms):
            entry = dict(p)
            entry["final_score"] = self.alpha * dn
            entry["source"]      = "dense_seed"
            results.append(entry)

        for p, gn in zip(expanded, graph_norms):
            entry = dict(p)
            entry["final_score"] = (1.0 - self.alpha) * gn
            entry["source"]      = "graph_expanded"
            results.append(entry)

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    def score_breakdown(
        self,
        seeds: list[dict],
        expanded: list[dict],
    ) -> list[dict]:
        """
        Return the full ranked list (no top_k cutoff) with per-passage
        score diagnostics. Useful for ablation analysis.

        Each entry adds:
            final_score   (float)
            dense_norm    (float) — normalized dense score (0 for expanded)
            graph_norm    (float) — normalized graph score (0 for seeds)
            source        (str)
        """
        dense_norms = _minmax_normalize([p["score"] for p in seeds])
        graph_norms = _minmax_normalize([p["score"] for p in expanded]) if expanded else []

        results = []
        for p, dn in zip(seeds, dense_norms):
            entry = dict(p)
            entry["dense_norm"]  = dn
            entry["graph_norm"]  = 0.0
            entry["final_score"] = self.alpha * dn
            entry["source"]      = "dense_seed"
            results.append(entry)

        for p, gn in zip(expanded, graph_norms):
            entry = dict(p)
            entry["dense_norm"]  = 0.0
            entry["graph_norm"]  = gn
            entry["final_score"] = (1.0 - self.alpha) * gn
            entry["source"]      = "graph_expanded"
            results.append(entry)

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
