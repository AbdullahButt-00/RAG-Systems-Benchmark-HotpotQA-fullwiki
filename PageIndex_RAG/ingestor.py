"""
Upload PageIndex document bundles and persist doc_ids.

Run once after preprocess.py:
    python ingestor.py
"""

import json
import time

from pathlib import Path
from tqdm import tqdm

from pageindex import PageIndexClient

from config import (
    PAGEINDEX_API_KEY,
    PAGEINDEX_POLL_INTERVAL,
    PAGEINDEX_DOCS_DIR,
    PAGEINDEX_INDEX_DIR,
    DOC_ID_PERSIST_PATH,
    PASSAGE_LOOKUP_PATH,
)


class PageIndexIngestor:
    """
    Uploads all bundle files from pageindex_docs/ to PageIndex and persists
    the bundle_filename → doc_id mapping to doc_ids.json.

    Idempotent: bundles already present in doc_ids.json are skipped.
    """

    def __init__(self) -> None:
        self._client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        PAGEINDEX_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    def _load_existing(self) -> dict:
        if DOC_ID_PERSIST_PATH.exists():
            with open(DOC_ID_PERSIST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self, mapping: dict) -> None:
        with open(DOC_ID_PERSIST_PATH, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

    def _poll_until_ready(self, doc_id: str) -> None:
        """Block until the document status is 'completed' or raise on 'failed'."""
        while True:
            info = self._client.get_document(doc_id)
            status = info.get("status", "")
            if status == "completed":
                return
            if status == "failed":
                raise RuntimeError(f"PageIndex processing failed for doc_id={doc_id}")
            time.sleep(PAGEINDEX_POLL_INTERVAL)

    def upload_all(self) -> dict:
        """
        Upload every bundle file that does not yet have a doc_id.

        Returns the complete bundle_filename → doc_id mapping.
        """
        if not PAGEINDEX_DOCS_DIR.exists():
            raise FileNotFoundError(
                f"Bundle directory not found: {PAGEINDEX_DOCS_DIR}\n"
                "Run preprocess.py first."
            )

        bundle_files = sorted(PAGEINDEX_DOCS_DIR.glob("bundle_*.pdf"))
        if not bundle_files:
            raise FileNotFoundError(
                f"No bundle files found in {PAGEINDEX_DOCS_DIR}.\n"
                "Run preprocess.py first."
            )

        mapping = self._load_existing()
        already_done = len([f for f in bundle_files if f.name in mapping])
        remaining = [f for f in bundle_files if f.name not in mapping]

        print(f"Total bundles : {len(bundle_files):,}")
        print(f"Already uploaded: {already_done:,}")
        print(f"To upload     : {len(remaining):,}")

        if not remaining:
            print("All bundles already uploaded.")
            return mapping

        for bundle_path in tqdm(remaining, desc="Uploading bundles", unit="bundle"):
            result = self._client.submit_document(str(bundle_path))
            doc_id = result["doc_id"]
            self._poll_until_ready(doc_id)
            mapping[bundle_path.name] = doc_id
            # Persist after every successful upload so partial runs are resumable
            self._save(mapping)

        print(f"\nAll bundles uploaded. doc_ids persisted → {DOC_ID_PERSIST_PATH}")
        return mapping

    def load_doc_ids(self) -> dict:
        """
        Load and return the persisted bundle_filename → doc_id mapping.

        Raises a clear error if ingestor has not been run yet.
        """
        if not DOC_ID_PERSIST_PATH.exists():
            raise FileNotFoundError(
                f"doc_ids.json not found at {DOC_ID_PERSIST_PATH}.\n"
                "Run ingestor.py first: python ingestor.py"
            )
        with open(DOC_ID_PERSIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_doc_ids(self) -> list[str]:
        """Return a flat list of all doc_ids (order matches bundle sort order)."""
        mapping = self.load_doc_ids()
        return [mapping[k] for k in sorted(mapping.keys())]

    def sync_passage_lookup(self) -> None:
        """
        Back-fill the doc_id field in passage_lookup.json from the persisted mapping.
        Called automatically at the end of upload_all.
        """
        if not PASSAGE_LOOKUP_PATH.exists():
            return

        mapping = self.load_doc_ids()

        with open(PASSAGE_LOOKUP_PATH, "r", encoding="utf-8") as f:
            lookup: dict = json.load(f)

        updated = 0
        for pid, entry in lookup.items():
            bundle_name = entry.get("bundle_file")
            if bundle_name and bundle_name in mapping:
                if entry.get("doc_id") != mapping[bundle_name]:
                    entry["doc_id"] = mapping[bundle_name]
                    updated += 1

        with open(PASSAGE_LOOKUP_PATH, "w", encoding="utf-8") as f:
            json.dump(lookup, f, ensure_ascii=False)

        print(f"passage_lookup.json updated: {updated:,} entries back-filled with doc_id.")


if __name__ == "__main__":
    ingestor = PageIndexIngestor()
    ingestor.upload_all()
    ingestor.sync_passage_lookup()
    print("\nIngestion complete.")
    print("Next step: python run_evaluation.py")
