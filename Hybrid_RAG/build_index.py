"""
One-shot build script: corpus → FAISS index → leakage-free passage graph.

Run this once before any evaluation. Estimated total time: ~60–90 min.

    python build_index.py

Steps
-----
  Step 1: build corpus.json        (~5 min)
  Step 2: build FAISS index        (~15–30 min, batch-streamed)
  Step 3: build graph_no_context   (~45–60 min)

Shortcut if Graph_RAG is already built
---------------------------------------
The corpus format and FAISS index are identical to Graph_RAG.
You can hard-link or copy those artifacts to skip Steps 1–2:

    mkdir -p data faiss_index
    ln -s ../Graph_RAG/data/corpus.json        data/corpus.json
    cp -r ../Graph_RAG/storage                 storage
    cp -r ../Graph_RAG/faiss_index             faiss_index

Then run only Step 3:

    python build_index.py --skip_corpus --skip_faiss

The graph MUST be rebuilt because the hybrid graph excludes same_context
edges that are present in Graph_RAG's graph.pkl.
"""

import argparse
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all artifacts for Confidence-Gated Hybrid-RAG"
    )
    parser.add_argument(
        "--skip_corpus",
        action="store_true",
        help="Skip corpus.json build (use if already symlinked from Graph_RAG)",
    )
    parser.add_argument(
        "--skip_faiss",
        action="store_true",
        help="Skip FAISS index build (use if already symlinked from Graph_RAG)",
    )
    parser.add_argument(
        "--skip_graph",
        action="store_true",
        help="Skip graph build (only useful for re-running corpus/FAISS alone)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.perf_counter()

    from config import CORPUS_PATH, FAISS_INDEX_FILE, STORAGE_DIR, GRAPH_PERSIST_PATH

    # ── Step 1: Corpus ─────────────────────────────────────────────────────
    if args.skip_corpus:
        print(f"[Step 1] Skipping corpus build (--skip_corpus).")
        if not CORPUS_PATH.exists():
            print(f"  ERROR: corpus.json not found at {CORPUS_PATH}.", file=sys.stderr)
            print(  "  Either run without --skip_corpus or symlink from Graph_RAG.", file=sys.stderr)
            sys.exit(1)
        print(f"  corpus.json found at {CORPUS_PATH}.")
    else:
        if CORPUS_PATH.exists():
            print(f"[Step 1] Corpus already exists — skipping rebuild.")
            print(f"  {CORPUS_PATH}")
        else:
            print("[Step 1] Building corpus.json...")
            t0 = time.perf_counter()
            from preprocess import build_corpus
            build_corpus()
            print(f"  Done in {(time.perf_counter()-t0)/60:.1f} min.")

    # ── Step 2: FAISS index ────────────────────────────────────────────────
    if args.skip_faiss:
        print(f"\n[Step 2] Skipping FAISS build (--skip_faiss).")
        if not FAISS_INDEX_FILE.exists() or not STORAGE_DIR.exists():
            print(f"  ERROR: FAISS index not found.", file=sys.stderr)
            sys.exit(1)
        print(f"  FAISS index found at {FAISS_INDEX_FILE}.")
    else:
        if FAISS_INDEX_FILE.exists() and STORAGE_DIR.exists():
            print(f"\n[Step 2] FAISS index already exists — skipping rebuild.")
            print(f"  {FAISS_INDEX_FILE}")
        else:
            print("\n[Step 2] Building FAISS index (batch-streaming, ~15–30 min)...")
            t0 = time.perf_counter()
            from retriever import DenseRetriever
            dense = DenseRetriever()
            dense.build_index(CORPUS_PATH)
            print(f"  Done in {(time.perf_counter()-t0)/60:.1f} min.")

    # ── Step 3: Leakage-free graph ─────────────────────────────────────────
    if args.skip_graph:
        print(f"\n[Step 3] Skipping graph build (--skip_graph).")
    else:
        if GRAPH_PERSIST_PATH.exists():
            print(f"\n[Step 3] Graph already exists — skipping rebuild.")
            print(f"  {GRAPH_PERSIST_PATH}")
        else:
            print("\n[Step 3] Building leakage-free passage graph (~45–60 min)...")
            print("  Edge types: title_link + entity_overlap  (no same_context)")
            t0 = time.perf_counter()
            from graph_builder import build_graph, save_graph
            g = build_graph()
            save_graph(g)
            print(f"  Done in {(time.perf_counter()-t0)/60:.1f} min.")

    elapsed = (time.perf_counter() - t_total) / 60
    print(f"\nAll artifacts built. Total time: {elapsed:.1f} min.")
    print("Next step: python run_evaluation.py")


if __name__ == "__main__":
    main()
