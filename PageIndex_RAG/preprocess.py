"""
Build the passage corpus and PageIndex document bundles for PageIndex-RAG.

Run once before ingestor.py:
    python preprocess.py
"""

import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

from config import (
    DATASET_DIR,
    DATA_DIR,
    CORPUS_PATH,
    PASSAGE_LOOKUP_PATH,
    PAGEINDEX_DOCS_DIR,
    PASSAGES_PER_DOCUMENT,
    CORPUS_MAX_PASSAGES,
)


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
        corpus.json  — passage_id → {title, text}
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
            if CORPUS_MAX_PASSAGES and counter >= CORPUS_MAX_PASSAGES:
                break
            titles = example["context"]["title"]
            sentences_list = example["context"]["sentences"]
            for title, sentences in zip(titles, sentences_list):
                if title in seen_titles:
                    continue
                text = title + ": " + " ".join(sentences)
                pid = str(counter)
                corpus[pid] = {"title": title, "text": text}
                seen_titles[title] = pid
                counter += 1
                if CORPUS_MAX_PASSAGES and counter >= CORPUS_MAX_PASSAGES:
                    break

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    print(f"\nCorpus saved: {len(corpus):,} unique passages → {CORPUS_PATH}")
    return corpus


def build_bundles(corpus: dict) -> dict:
    """
    Group passages into plain-text bundle files (PASSAGES_PER_DOCUMENT each).
    Each bundle file is formatted so PageIndex can be searched and cited back.

    Each passage block inside a bundle:
        [PASSAGE_ID: {pid}]
        [TITLE: {title}]
        {text}
        ---

    Saves:
        pageindex_docs/bundle_XXXX.txt  — one file per group
        passage_lookup.json             — passage_id → {title, text, bundle_file, doc_id}
                                          (doc_id filled later by ingestor.py)
    """
    PAGEINDEX_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    pids = list(corpus.keys())
    total = len(pids)
    num_bundles = (total + PASSAGES_PER_DOCUMENT - 1) // PASSAGES_PER_DOCUMENT

    passage_lookup: dict[str, dict] = {}

    print(f"Building {num_bundles:,} bundle files "
          f"({PASSAGES_PER_DOCUMENT} passages each)...")

    for bundle_idx in tqdm(range(num_bundles), desc="  bundles", unit="bundle"):
        start = bundle_idx * PASSAGES_PER_DOCUMENT
        end = min(start + PASSAGES_PER_DOCUMENT, total)
        batch_pids = pids[start:end]

        bundle_name = f"bundle_{bundle_idx:05d}.txt"
        bundle_path = PAGEINDEX_DOCS_DIR / bundle_name

        lines = []
        for pid in batch_pids:
            entry = corpus[pid]
            lines.append(f"[PASSAGE_ID: {pid}]")
            lines.append(f"[TITLE: {entry['title']}]")
            lines.append(entry["text"])
            lines.append("---")
            lines.append("")  # blank line between blocks

            passage_lookup[pid] = {
                "title": entry["title"],
                "text": entry["text"],
                "bundle_file": bundle_name,
                "doc_id": None,  # filled by ingestor.py after upload
            }

        bundle_path.write_text("\n".join(lines), encoding="utf-8")

    with open(PASSAGE_LOOKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(passage_lookup, f, ensure_ascii=False)

    print(f"Bundle files saved → {PAGEINDEX_DOCS_DIR}")
    print(f"Passage lookup saved → {PASSAGE_LOOKUP_PATH}")
    return passage_lookup


if __name__ == "__main__":
    if CORPUS_PATH.exists():
        print(f"Corpus already exists at {CORPUS_PATH}. Loading...")
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        print(f"Loaded {len(corpus):,} passages.")
    else:
        corpus = build_corpus()

    # Count existing bundles to decide whether to rebuild
    existing_bundles = list(PAGEINDEX_DOCS_DIR.glob("bundle_*.txt")) if PAGEINDEX_DOCS_DIR.exists() else []
    if existing_bundles and PASSAGE_LOOKUP_PATH.exists():
        print(f"\nBundle files already exist ({len(existing_bundles):,} files). Skipping rebuild.")
        print(f"  {PAGEINDEX_DOCS_DIR}")
        print(f"  {PASSAGE_LOOKUP_PATH}")
    else:
        build_bundles(corpus)

    print("\nPreprocessing complete.")
    print("Next step: python ingestor.py")


def build_bundles_pdf(corpus: dict) -> dict:
    """Rebuild bundles as reportlab PDFs (PageIndex requires PDF)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet

    PAGEINDEX_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    pids = list(corpus.keys())
    total = len(pids)
    num_bundles = (total + PASSAGES_PER_DOCUMENT - 1) // PASSAGES_PER_DOCUMENT
    passage_lookup: dict[str, dict] = {}

    print(f"Building {num_bundles:,} PDF bundles ({PASSAGES_PER_DOCUMENT} passages each)...")

    for bundle_idx in tqdm(range(num_bundles), desc="  bundles", unit="bundle"):
        start = bundle_idx * PASSAGES_PER_DOCUMENT
        end = min(start + PASSAGES_PER_DOCUMENT, total)
        batch_pids = pids[start:end]

        bundle_name = f"bundle_{bundle_idx:05d}.pdf"
        bundle_path = PAGEINDEX_DOCS_DIR / bundle_name

        story = []
        for pid in batch_pids:
            entry = corpus[pid]
            line = entry["text"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            story.append(Paragraph(line, styles["Normal"]))

            passage_lookup[pid] = {
                "title": entry["title"],
                "text": entry["text"],
                "bundle_file": bundle_name,
                "doc_id": None,
            }

        doc = SimpleDocTemplate(str(bundle_path), pagesize=A4)
        doc.build(story)

    with open(PASSAGE_LOOKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(passage_lookup, f, ensure_ascii=False)

    print(f"PDF bundles saved → {PAGEINDEX_DOCS_DIR}")
    print(f"Passage lookup saved → {PASSAGE_LOOKUP_PATH}")
    return passage_lookup
