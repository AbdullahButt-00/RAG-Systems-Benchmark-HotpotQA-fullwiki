"""
Graph-guided multi-hop retriever.

Seeds from FAISS dense retrieval, then expands via BFS over the passage graph.
The final ranking combines dense similarity scores for seeds with
edge-weight / hop-distance decay scores for expanded nodes.
"""

from collections import deque

from config import TOP_K_SEED, TOP_K_GRAPH, MAX_HOP


class GraphRetriever:
    """
    Combines dense seed retrieval with passage-graph BFS expansion.

    Parameters
    ----------
    dense_retriever : DenseRetriever
        Must already have load_index() called.
    graph : nx.DiGraph
        Loaded NetworkX passage graph (from graph_builder.load_graph()).
    corpus : dict
        passage_id → {title, text, sentences}  (loaded from corpus.json)
    """

    def __init__(self, dense_retriever, graph, corpus: dict) -> None:
        self._dense = dense_retriever
        self._graph = graph
        self._corpus = corpus

    def retrieve(
        self,
        query: str,
        top_k_seed: int = TOP_K_SEED,
        top_k_graph: int = TOP_K_GRAPH,
        max_hop: int = MAX_HOP,
    ) -> list[dict]:
        """
        Full graph-guided retrieval for a single query.

        Step 1 — SEED   : dense retrieval → top_k_seed passage_ids
        Step 2 — EXPAND : BFS on passage graph up to max_hop hops
        Step 3 — RANK   : sort by score (dense score for seeds,
                           edge_weight / (hop+2) for expanded)
        Step 4 — RETURN : top_k_graph passages with full metadata

        Returns
        -------
        list[dict] with keys:
            passage_id  (str)
            title       (str)
            text        (str)
            sentences   (list[str])
            score       (float)
            hop_distance (int)   0 = seed
            is_seed     (bool)
        """
        # ── Step 1: Dense seed retrieval ──────────────────────────────────
        seeds = self._dense.retrieve_seed(query, top_k=top_k_seed)

        # ── Step 2: BFS graph expansion ───────────────────────────────────
        candidates: dict[str, dict] = {}
        visited: set[str] = set()
        queue: deque = deque()

        for s in seeds:
            pid = s["passage_id"]
            if not pid:
                continue
            visited.add(pid)
            candidates[pid] = {
                "passage_id": pid,
                "score": s["score"],
                "hop_distance": 0,
                "is_seed": True,
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
                weight = edge_data.get("weight", 0.5)
                # Score decays with each hop; anchored to edge weight.
                score = weight / (hop + 2)

                candidates[neighbor] = {
                    "passage_id": neighbor,
                    "score": score,
                    "hop_distance": hop + 1,
                    "is_seed": False,
                }
                queue.append((neighbor, hop + 1, score))

        # ── Step 3: Rank by score descending ──────────────────────────────
        ranked = sorted(
            candidates.values(),
            key=lambda x: x["score"],
            reverse=True,
        )
        top = ranked[:top_k_graph]

        # ── Step 4: Attach text + metadata from corpus ────────────────────
        for p in top:
            entry = self._corpus.get(p["passage_id"], {})
            p["title"] = entry.get("title", "")
            p["text"] = entry.get("text", "")
            p["sentences"] = entry.get("sentences", [])

        return top
