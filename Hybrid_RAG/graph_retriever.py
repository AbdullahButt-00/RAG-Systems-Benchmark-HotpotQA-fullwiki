"""
Graph-guided BFS expansion retriever for Hybrid-RAG.

Key difference from Graph_RAG/graph_retriever.py
-------------------------------------------------
Adds expand_from_seeds() method that accepts pre-computed seed passages
and performs only the BFS stage, returning expanded (non-seed) passages.

This avoids a redundant second FAISS call in the hybrid pipeline:
  - Stage 1 already ran dense retrieval → seeds
  - Stage 3 calls expand_from_seeds(seeds) → expanded
  - No duplicate embedding lookup

The original retrieve() method is preserved for standalone use.
"""

from collections import deque

from config import TOP_K_SEED, TOP_K_GRAPH, MAX_HOP


class GraphRetriever:
    """
    BFS-based passage expansion over the leakage-free passage graph.

    Parameters
    ----------
    dense_retriever : DenseRetriever
        Must have load_index() already called.
    graph : nx.DiGraph
        Loaded passage graph (from graph_builder.load_graph()).
    corpus : dict
        passage_id → {title, text, sentences}  (loaded from corpus.json)
    """

    def __init__(self, dense_retriever, graph, corpus: dict) -> None:
        self._dense  = dense_retriever
        self._graph  = graph
        self._corpus = corpus

    # ------------------------------------------------------------------
    # Primary hybrid interface
    # ------------------------------------------------------------------

    def expand_from_seeds(
        self,
        seeds: list[dict],
        top_k_graph: int = TOP_K_GRAPH,
        max_hop: int = MAX_HOP,
    ) -> list[dict]:
        """
        BFS expansion from pre-computed seed passages.

        Takes seeds already retrieved by DenseRetriever (Stage 1) and
        expands outward over the passage graph without re-running dense
        retrieval. Returns only the newly discovered (non-seed) passages.

        Scoring
        -------
        Each expanded node's score is:
            score = edge_weight / (hop + 2)
        This decays with hop distance and is proportional to edge strength.
        Hop 1: score in (0.08, 0.50) depending on edge type.
        Hop 2: score in (0.04, 0.25).

        Parameters
        ----------
        seeds : list[dict]
            Output of DenseRetriever.retrieve_seed(). Must have 'passage_id' key.
        top_k_graph : int
            Maximum number of expanded (non-seed) passages to return.
        max_hop : int
            BFS depth limit.

        Returns
        -------
        list[dict] with keys:
            passage_id   (str)
            title        (str)
            text         (str)
            sentences    (list[str])
            score        (float)   — edge_weight / (hop + 2)
            hop_distance (int)     — 1 or 2
            is_seed      (bool)    — always False
        """
        visited: set[str] = set()
        queue:   deque    = deque()

        for s in seeds:
            pid = s.get("passage_id", "")
            if not pid:
                continue
            # Mark seed as visited so it cannot appear in candidates.
            visited.add(pid)
            if pid in self._graph:
                queue.append((pid, 0, s.get("score", 0.0)))

        candidates: dict[str, dict] = {}

        while queue:
            pid, hop, _ = queue.popleft()
            if hop >= max_hop:
                continue

            for neighbor in self._graph.successors(pid):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                edge_data = self._graph[pid][neighbor]
                weight    = edge_data.get("weight", 0.5)
                score     = weight / (hop + 2)

                candidates[neighbor] = {
                    "passage_id":   neighbor,
                    "score":        score,
                    "hop_distance": hop + 1,
                    "is_seed":      False,
                }
                queue.append((neighbor, hop + 1, score))

        # candidates contains only non-seed passages by construction:
        # all seed_pids were added to `visited` before the BFS loop, so
        # they can never pass the `if neighbor in visited: continue` guard.
        ranked = sorted(
            candidates.values(),
            key=lambda x: x["score"],
            reverse=True,
        )
        top = ranked[:top_k_graph]

        # Attach text + metadata from corpus
        for p in top:
            entry = self._corpus.get(p["passage_id"], {})
            p["title"]     = entry.get("title", "")
            p["text"]      = entry.get("text", "")
            p["sentences"] = entry.get("sentences", [])

        return top

    # ------------------------------------------------------------------
    # Standalone interface (compatible with Graph_RAG's GraphRetriever)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k_seed: int = TOP_K_SEED,
        top_k_graph: int = TOP_K_GRAPH,
        max_hop: int = MAX_HOP,
    ) -> list[dict]:
        """
        Full graph-guided retrieval: dense seeds + BFS expansion combined.

        Kept for compatibility and standalone testing.
        The hybrid pipeline uses expand_from_seeds() instead to avoid
        running dense retrieval twice.

        Returns
        -------
        list[dict] with keys:
            passage_id, title, text, sentences, score, hop_distance, is_seed
        """
        seeds = self._dense.retrieve_seed(query, top_k=top_k_seed)

        candidates: dict[str, dict] = {}
        visited:    set[str] = set()
        queue:      deque    = deque()

        for s in seeds:
            pid = s["passage_id"]
            if not pid:
                continue
            visited.add(pid)
            candidates[pid] = {
                "passage_id":   pid,
                "score":        s["score"],
                "hop_distance": 0,
                "is_seed":      True,
            }
            if pid in self._graph:
                queue.append((pid, 0, s["score"]))

        while queue:
            pid, hop, _ = queue.popleft()
            if hop >= max_hop:
                continue
            for neighbor in self._graph.successors(pid):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                edge_data = self._graph[pid][neighbor]
                weight    = edge_data.get("weight", 0.5)
                score     = weight / (hop + 2)
                candidates[neighbor] = {
                    "passage_id":   neighbor,
                    "score":        score,
                    "hop_distance": hop + 1,
                    "is_seed":      False,
                }
                queue.append((neighbor, hop + 1, score))

        ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
        top    = ranked[:top_k_graph]

        for p in top:
            entry = self._corpus.get(p["passage_id"], {})
            p["title"]     = entry.get("title", "")
            p["text"]      = entry.get("text", "")
            p["sentences"] = entry.get("sentences", [])

        return top
