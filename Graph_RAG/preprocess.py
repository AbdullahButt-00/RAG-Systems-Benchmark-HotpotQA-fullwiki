"""
Build the passage corpus, FAISS index, and context clusters for Graph-RAG.

Run once before graph_builder.py:
    python preprocess.py
"""

import json
from pathlib import Path

from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from llama_index.core import Document

from config import DATASET_DIR, DATA_DIR, CORPUS_PATH, CONTEXT_CLUSTERS_PATH, CORPUS_MAX_PASSAGES


def _find_arrow_dir() -> Path:
    """
    Locate the Arrow data directory inside the HuggingFace cache structure:
        DATASET/hotpotqa___hotpot_qa/fullwiki/0.0.0/<hash>/
    Works regardless of the specific hash value.
    """
    version_dir = DATASET_DIR / "hotpotqa___hotpot_qa" / "fullwiki" / "0.0.0"
    if not version_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset directory not found: {version_dir}\n"
            "Make sure the HotpotQA dataset is in DATASET/."
        )
    candidates = [d for d in version_dir.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No data sub-directory found inside {version_dir}.")
    return candidates[0]


def _load_splits() -> dict:
    """
    Load train and validation splits directly from Arrow files,
    bypassing HuggingFace cache validation (avoids .incomplete_info.lock issues).
    """
    arrow_dir = _find_arrow_dir()
    print(f"Loading Arrow files from: {arrow_dir}")
    train = concatenate_datasets([
        Dataset.from_file(str(arrow_dir / "hotpot_qa-train-00000-of-00002.arrow")),
        Dataset.from_file(str(arrow_dir / "hotpot_qa-train-00001-of-00002.arrow")),
    ])
    validation = Dataset.from_file(str(arrow_dir / "hotpot_qa-validation.arrow"))
    return {"train": train, "validation": validation}


def build_corpus() -> dict:
    """
    Load HotpotQA train+validation, deduplicate passages by title.

    Saves:
        corpus.json           — passage_id → {title, text, sentences}
        context_clusters.json — list of [passage_id, ...] per HotpotQA example
                                (used by graph_builder for same_context edges)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading HotpotQA dataset from local Arrow files...")
    dataset = _load_splits()

    corpus: dict[str, dict] = {}        # pid → {title, text, sentences}
    seen_titles: dict[str, str] = {}    # title → pid (deduplication)
    clusters: list[list[str]] = []      # one list of pids per question
    counter = 0

    for split_name in ("train", "validation"):
        split = dataset[split_name]
        print(f"Processing {split_name} split ({len(split):,} examples)...")
        for example in tqdm(split, desc=f"  {split_name}", unit="ex"):
            if CORPUS_MAX_PASSAGES and counter >= CORPUS_MAX_PASSAGES:
                break
            titles = example["context"]["title"]
            sentences_list = example["context"]["sentences"]
            cluster_pids: list[str] = []

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
                    if CORPUS_MAX_PASSAGES and counter >= CORPUS_MAX_PASSAGES:
                        break
                cluster_pids.append(seen_titles[title])

            clusters.append(cluster_pids)

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    with open(CONTEXT_CLUSTERS_PATH, "w", encoding="utf-8") as f:
        json.dump(clusters, f)

    print(f"\nCorpus saved   : {len(corpus):,} unique passages → {CORPUS_PATH}")
    print(f"Clusters saved : {len(clusters):,} examples      → {CONTEXT_CLUSTERS_PATH}")
    return corpus


def load_documents() -> list[Document]:
    """
    Read corpus.json and return a list of LlamaIndex Document objects.
    Each Document carries title, passage_id, and sentences list in metadata
    (sentences needed for Support Coverage metric at evaluation time).
    """
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
            # Keep metadata minimal — LlamaIndex serialises it into every node
            # and raises ValueError if len(metadata_str) > chunk_size.
            # Sentences are retrieved at query time from corpus.json via passage_id.
            metadata={
                "title": entry["title"],
                "passage_id": pid,
            },
        )
        for pid, entry in corpus.items()
    ]
    print(f"Loaded {len(documents):,} documents.")
    return documents


if __name__ == "__main__":
    if CORPUS_PATH.exists() and CONTEXT_CLUSTERS_PATH.exists():
        print("Corpus and clusters already exist. Skipping rebuild.")
        print(f"  {CORPUS_PATH}")
        print(f"  {CONTEXT_CLUSTERS_PATH}")
    else:
        build_corpus()

    from retriever import DenseRetriever

    # Pass the corpus path — build_index streams in batches, never loading
    # all 508k Document objects at once.
    retriever = DenseRetriever()
    retriever.build_index(CORPUS_PATH)
    print("\nPreprocessing complete.")
    print("Next step: python graph_builder.py")
