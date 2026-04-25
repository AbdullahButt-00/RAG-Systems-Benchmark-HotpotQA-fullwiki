"""
Build and persist the leakage-free passage graph for Hybrid-RAG.

Edge types
----------
  title_link     (weight 1.0) — directed: P→Q if Q's title appears in P's text
  entity_overlap (weight 0.1–1.0) — undirected: passages sharing ≥1 named-entity span

Intentionally excluded
----------------------
  same_context edges ARE NOT built here.
  In Graph_RAG, same_context edges connect passages that co-appear in the
  same HotpotQA question's context field. This encodes dataset structure
  (i.e., which passages the dataset paired together) directly into the graph,
  creating a data-leakage risk. Excluding them makes the hybrid graph
  generalizable beyond HotpotQA.

  To quantify the leakage effect, use Graph_RAG's original graph.pkl
  (which includes same_context edges) as ablation A6.

Run order
---------
    python preprocess.py    # corpus.json must exist first
    python graph_builder.py # builds graph_no_context.pkl (~45 min)
"""

import json
import pickle
import re
from collections import defaultdict
from itertools import combinations

import networkx as nx
from tqdm import tqdm

from config import (
    CORPUS_PATH,
    GRAPH_PERSIST_PATH,
    SHARED_ENTITY_THRESHOLD,
)

# Matches capitalized multi-word spans: "Barack Obama", "New York City", etc.
_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# Prevents "United States" from generating O(n²) pairs across all passages.
_MAX_PASSAGES_PER_ENTITY = 50


def _load_corpus() -> dict:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Edge type 1: Title links
# ---------------------------------------------------------------------------

def _build_title_links(corpus: dict, graph: nx.DiGraph) -> int:
    """
    Add directed edge P → Q if passage Q's title appears verbatim in passage P's text.

    Uses an inverted index on capitalized first-words of titles to avoid O(n²) scan.
    Returns number of edges added.
    """
    print("  Building title-link edges...")

    first_word_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pid, entry in corpus.items():
        parts = entry["title"].split()
        if parts and parts[0][0].isupper():
            fw = parts[0].lower()
            first_word_index[fw].append((entry["title"].lower(), pid))

    edge_count = 0
    for pid_p, entry_p in tqdm(corpus.items(), desc="    Title links", unit="p", mininterval=2):
        text_lower = entry_p["text"].lower()
        cap_words = {
            w.strip(".,;:!?\"'()[]").lower()
            for w in entry_p["text"].split()
            if w and w[0].isupper()
        }
        for fw in cap_words:
            for title_lower, pid_q in first_word_index.get(fw, []):
                if pid_q == pid_p:
                    continue
                if title_lower in text_lower:
                    if not graph.has_edge(pid_p, pid_q):
                        graph.add_edge(pid_p, pid_q, edge_type="title_link", weight=1.0)
                        edge_count += 1
    return edge_count


# ---------------------------------------------------------------------------
# Edge type 2: Entity overlap
# ---------------------------------------------------------------------------

def _build_entity_edges(
    corpus: dict,
    graph: nx.DiGraph,
    threshold: int = SHARED_ENTITY_THRESHOLD,
) -> int:
    """
    Add undirected entity-overlap edges between passages sharing ≥ threshold
    capitalized multi-word phrases.

    Skips entities appearing in fewer than 2 or more than _MAX_PASSAGES_PER_ENTITY
    passages to control edge explosion.
    Returns number of directed edge slots added (each undirected edge = 2).
    """
    print("  Building entity-overlap edges...")

    entity_to_pids: dict[str, set] = defaultdict(set)
    pid_to_entities: dict[str, set] = {}

    for pid, entry in tqdm(corpus.items(), desc="    Extracting entities", unit="p", mininterval=2):
        entities = set(_ENTITY_RE.findall(entry["text"]))
        pid_to_entities[pid] = entities
        for ent in entities:
            entity_to_pids[ent].add(pid)

    edge_count = 0
    for ent, pids in tqdm(
        entity_to_pids.items(),
        desc="    Linking entity pairs",
        unit="ent",
        mininterval=2,
    ):
        pids_list = list(pids)
        n = len(pids_list)
        if n < 2 or n > _MAX_PASSAGES_PER_ENTITY:
            continue

        for pid_a, pid_b in combinations(pids_list, 2):
            for src, dst in ((pid_a, pid_b), (pid_b, pid_a)):
                if graph.has_edge(src, dst):
                    data = graph[src][dst]
                    if data.get("edge_type") == "entity_overlap":
                        data["_cnt"] = data.get("_cnt", 1) + 1
                else:
                    graph.add_edge(src, dst, edge_type="entity_overlap", _cnt=1, weight=0.1)
                    edge_count += 1

    # Normalize entity-overlap weights: weight = min(1.0, shared_count / 5)
    for src, dst, data in graph.edges(data=True):
        if data.get("edge_type") == "entity_overlap":
            cnt = data.pop("_cnt", 1)
            data["weight"] = min(1.0, cnt / 5.0)

    return edge_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph() -> nx.DiGraph:
    """
    Build the leakage-free passage graph.

    Uses only title_link + entity_overlap edges.
    same_context edges are intentionally excluded (see module docstring).
    """
    print("Loading corpus...")
    corpus = _load_corpus()

    print(f"\nInitialising directed graph with {len(corpus):,} nodes...")
    graph: nx.DiGraph = nx.DiGraph()
    for pid, entry in corpus.items():
        graph.add_node(pid, title=entry["title"])

    print("\nBuilding edges (title_link + entity_overlap only):")
    n_title  = _build_title_links(corpus, graph)
    n_entity = _build_entity_edges(corpus, graph)

    n_nodes  = graph.number_of_nodes()
    n_edges  = graph.number_of_edges()
    avg_deg  = n_edges / n_nodes if n_nodes else 0.0
    density  = nx.density(graph)

    print(f"\nGraph statistics (leakage-free):")
    print(f"  Nodes           : {n_nodes:,}")
    print(f"  Total edges     : {n_edges:,}")
    print(f"  Title links     : {n_title:,}")
    print(f"  Entity overlaps : {n_entity:,}")
    print(f"  [same_context   : excluded — see graph_builder.py docstring]")
    print(f"  Avg out-degree  : {avg_deg:.2f}")
    print(f"  Density         : {density:.8f}")

    return graph


def save_graph(graph: nx.DiGraph) -> None:
    """Persist graph to disk using stdlib pickle."""
    GRAPH_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PERSIST_PATH, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = GRAPH_PERSIST_PATH.stat().st_size / 1_048_576
    print(f"Graph saved → {GRAPH_PERSIST_PATH}  ({size_mb:.1f} MB)")


def load_graph() -> nx.DiGraph:
    """Load graph from disk."""
    if not GRAPH_PERSIST_PATH.exists():
        raise FileNotFoundError(
            f"Graph not found at {GRAPH_PERSIST_PATH}.\n"
            "Build it first:  python graph_builder.py"
        )
    with open(GRAPH_PERSIST_PATH, "rb") as f:
        graph = pickle.load(f)
    print(
        f"Graph loaded: {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges  (no same_context edges)"
    )
    return graph


if __name__ == "__main__":
    if GRAPH_PERSIST_PATH.exists():
        print(f"Graph already exists at {GRAPH_PERSIST_PATH}.")
        print("Delete it manually to force a rebuild.")
    else:
        g = build_graph()
        save_graph(g)
    print("\nNext step: python run_evaluation.py")
