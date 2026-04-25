"""
Build the passage corpus and FAISS index for Hybrid-RAG.

Run order
---------
    python preprocess.py    # builds corpus.json + FAISS index
    python graph_builder.py # builds graph_no_context.pkl

Shortcut if Graph_RAG artifacts already exist
---------------------------------------------
The corpus format and FAISS index are identical to Graph_RAG.
You can hard-link or symlink to avoid a full rebuild (~1 hour):

    mkdir -p data faiss_index storage
    ln -s ../Graph_RAG/data/corpus.json        data/corpus.json
    ln -s ../Graph_RAG/faiss_index/index.faiss faiss_index/index.faiss
    cp -r ../Graph_RAG/storage                 storage

Then skip directly to:
    python graph_builder.py

Key difference from Graph_RAG/preprocess.py
--------------------------------------------
We do NOT build context_clusters.json here because the hybrid graph
uses only title_link + entity_overlap edges (no same_context edges).
Removing context_edges eliminates the dataset-structure leakage risk
present in Graph_RAG's graph.pkl.
"""

import json
from pathlib import Path

from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from llama_index.core import Document

from config import DATASET_DIR, DATA_DIR, CORPUS_PATH, FAISS_INDEX_FILE, STORAGE_DIR


# ---------------------------------------------------------------------------
# Arrow file helpers (identical to Graph_RAG)
# ---------------------------------------------------------------------------

def _find_arrow_dir() -> Path:
    """Locate the Arrow data directory inside the HuggingFace cache structure."""
    version_dir = DATASET_DIR / "hotpotqa___hotpot_qa" / "fullwiki" / "0.0.0"
    if not version_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset directory not found: {version_dir}\n"
            "Run dataset_download.py from the project root first."
        )
    candidates = [d for d in version_dir.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No data sub-directory found inside {version_dir}.")
    return candidates[0]


def _load_splits() -> dict:
    """
    Load train + validation splits from Arrow files.
    Bypasses HuggingFace cache validation to avoid .incomplete_info.lock issues.
    """
    arrow_dir = _find_arrow_dir()
    print(f"Loading Arrow files from: {arrow_dir}")
    train = concatenate_datasets([
        Dataset.from_file(str(arrow_dir / "hotpot_qa-train-00000-of-00002.arrow")),
        Dataset.from_file(str(arrow_dir / "hotpot_qa-train-00001-of-00002.arrow")),
    ])
    validation = Dataset.from_file(str(arrow_dir / "hotpot_qa-validation.arrow"))
    return {"train": train, "validation": validation}


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def build_corpus() -> dict:
    """
    Load HotpotQA train+validation, deduplicate passages by title.

    Saves corpus.json: passage_id → {title, text, sentences}

    Does NOT save context_clusters.json — the hybrid graph does not use
    same_context edges, so cluster data is never needed.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading HotpotQA dataset from local Arrow files...")
    dataset = _load_splits()

    corpus: dict[str, dict] = {}
    seen_titles: dict[str, str] = {}
    counter = 0

    for split_name in ("train", "validation"):
        split = dataset[split_name]
        print(f"Processing {split_name} split ({len(split):,} examples)...")
        for example in tqdm(split, desc=f"  {split_name}", unit="ex"):
            titles = example["context"]["title"]
            sentences_list = example["context"]["sentences"]

            for title, sentences in zip(titles, sentences_list):
                if title not in seen_titles:
                    text = title + ": " + " ".join(sentences)
                    pid = str(counter)
                    corpus[pid] = {
                        "title": title,
                        "text": text,
                        "sentences": list(sentences),
                    }
                    seen_titles[title] = pid
                    counter += 1

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    print(f"\nCorpus saved: {len(corpus):,} unique passages → {CORPUS_PATH}")
    return corpus


def load_documents() -> list[Document]:
    """Read corpus.json and return LlamaIndex Document objects (lightweight metadata)."""
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found at {CORPUS_PATH}. Run build_corpus() first."
        )
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus: dict = json.load(f)

    documents = [
        Document(
            text=entry["text"],
            doc_id=pid,
            # Keep metadata minimal — LlamaIndex serialises it into every node.
            # Sentences are fetched at query time from corpus.json via passage_id.
            metadata={"title": entry["title"], "passage_id": pid},
        )
        for pid, entry in corpus.items()
    ]
    print(f"Loaded {len(documents):,} documents.")
    return documents


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Corpus ───────────────────────────────────────────────────────────────
    if CORPUS_PATH.exists():
        print(f"Corpus already exists at {CORPUS_PATH}. Skipping rebuild.")
    else:
        build_corpus()

    # ── FAISS index ───────────────────────────────────────────────────────────
    if FAISS_INDEX_FILE.exists() and STORAGE_DIR.exists():
        print(f"FAISS index already exists at {FAISS_INDEX_FILE}. Skipping rebuild.")
    else:
        from retriever import DenseRetriever
        retriever = DenseRetriever()
        retriever.build_index(CORPUS_PATH)

    print("\nPreprocessing complete.")
    print("Next step: python graph_builder.py")
