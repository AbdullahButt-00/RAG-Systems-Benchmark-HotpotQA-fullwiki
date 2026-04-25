"""
PageIndex retriever: wraps chat_completions API with Option B pre-filtering.

At startup, builds an inverted index: title_keyword → [doc_ids] from
passage_lookup.json. At query time, selects up to MAX_CANDIDATE_DOCS
candidate doc_ids whose bundles contain passages matching question keywords,
then queries only those doc_ids.
"""

import json
import re
import string
from collections import defaultdict

from pageindex import PageIndexClient

from config import (
    PAGEINDEX_API_KEY,
    PAGEINDEX_ENABLE_CITATIONS,
    PAGEINDEX_TEMPERATURE,
    MAX_CANDIDATE_DOCS,
    PASSAGE_LOOKUP_PATH,
)


def _tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if len(t) > 2]


_STOPWORDS = frozenset({
    "the", "and", "was", "are", "for", "that", "this", "with", "from",
    "his", "her", "their", "have", "has", "had", "not", "but", "what",
    "which", "who", "whom", "when", "where", "how", "did", "does", "been",
    "were", "its", "also", "into", "than", "then", "they", "them", "some",
    "can", "all", "would", "could", "should", "will", "may", "might",
})


class PageIndexRetriever:
    """
    Wraps PageIndex chat_completions to act as both retriever and reader.

    Option B pre-filter
    -------------------
    At __init__ time, builds an inverted index:
        keyword → set of doc_ids
    from passage_lookup.json (title tokens as keywords).

    At query time:
        1. Tokenize question, remove stopwords
        2. Look up each keyword → candidate doc_ids
        3. Score doc_ids by keyword hit count, take top MAX_CANDIDATE_DOCS
        4. Query only those doc_ids
        5. Merge answers: return the response with the most citations;
           fall back to the first non-empty answer if none have citations.
    """

    def __init__(self) -> None:
        self._client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        self._passage_lookup: dict = self._load_passage_lookup()
        # inverted index: keyword → sorted list of doc_ids (built once)
        self._keyword_index: dict[str, list[str]] = self._build_keyword_index()
        print(f"PageIndexRetriever ready. "
              f"{len(self._passage_lookup):,} passages indexed across "
              f"{len(set(e['doc_id'] for e in self._passage_lookup.values() if e.get('doc_id'))):,} documents.")

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _load_passage_lookup(self) -> dict:
        if not PASSAGE_LOOKUP_PATH.exists():
            raise FileNotFoundError(
                f"passage_lookup.json not found at {PASSAGE_LOOKUP_PATH}.\n"
                "Run preprocess.py then ingestor.py first."
            )
        with open(PASSAGE_LOOKUP_PATH, "r", encoding="utf-8") as f:
            lookup = json.load(f)
        # Verify doc_ids are populated
        missing = sum(1 for e in lookup.values() if not e.get("doc_id"))
        if missing:
            raise RuntimeError(
                f"{missing:,} passages have no doc_id in passage_lookup.json.\n"
                "Run ingestor.py to upload bundles and back-fill doc_ids."
            )
        return lookup

    def _build_keyword_index(self) -> dict[str, list[str]]:
        """Build keyword → [doc_id, ...] inverted index from passage titles."""
        index: dict[str, set] = defaultdict(set)
        for entry in self._passage_lookup.values():
            doc_id = entry["doc_id"]
            for token in _tokenize(entry["title"]):
                if token not in _STOPWORDS:
                    index[token].add(doc_id)
        return {k: sorted(v) for k, v in index.items()}

    # ------------------------------------------------------------------
    # Pre-filter
    # ------------------------------------------------------------------

    def _select_candidate_doc_ids(self, question: str) -> list[str]:
        """
        Score each doc_id by how many question keywords appear in its index entry.
        Return top MAX_CANDIDATE_DOCS doc_ids.
        """
        tokens = [t for t in _tokenize(question) if t not in _STOPWORDS]
        scores: dict[str, int] = defaultdict(int)
        for token in tokens:
            for doc_id in self._keyword_index.get(token, []):
                scores[doc_id] += 1

        if not scores:
            # No keyword hits — fall back to returning no candidates
            # (query will return empty result rather than querying all docs)
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:MAX_CANDIDATE_DOCS]]

    # ------------------------------------------------------------------
    # Citation parsing
    # ------------------------------------------------------------------

    def _parse_citations(self, citations: list, doc_id: str) -> list[dict]:
        """
        Map PageIndex citation objects back to passage_id + title using
        the structured markers in the bundle text.

        Each citation is expected to contain a text snippet. We search for
        [PASSAGE_ID: ...] and [TITLE: ...] markers within or near the snippet.
        Falls back to substring matching against passage_lookup text.
        """
        parsed = []
        pid_re = re.compile(r"\[PASSAGE_ID:\s*(\d+)\]")
        title_re = re.compile(r"\[TITLE:\s*(.+?)\]")

        # Build a doc_id → [pid] lookup for fast candidate filtering
        doc_passages = [
            (pid, entry)
            for pid, entry in self._passage_lookup.items()
            if entry.get("doc_id") == doc_id
        ]

        for idx, citation in enumerate(citations):
            cited_text = ""
            # PageIndex citation format may vary — try common fields
            if isinstance(citation, dict):
                cited_text = (
                    citation.get("text", "")
                    or citation.get("content", "")
                    or citation.get("excerpt", "")
                    or str(citation)
                )
            else:
                cited_text = str(citation)

            # Try to find PASSAGE_ID marker in the citation text
            pid_match = pid_re.search(cited_text)
            title_match = title_re.search(cited_text)

            if pid_match:
                pid = pid_match.group(1)
                entry = self._passage_lookup.get(pid)
                if entry:
                    parsed.append({
                        "passage_id": pid,
                        "title": entry["title"],
                        "text": entry["text"],
                        "citation_index": idx,
                    })
                    continue

            if title_match:
                title = title_match.group(1).strip()
                # Find pid by title
                for pid, entry in doc_passages:
                    if entry["title"] == title:
                        parsed.append({
                            "passage_id": pid,
                            "title": entry["title"],
                            "text": entry["text"],
                            "citation_index": idx,
                        })
                        break
                continue

            # Last resort: substring match on cited_text against passage texts
            best_pid, best_score = None, 0
            for pid, entry in doc_passages:
                overlap = len(set(cited_text.lower().split()) &
                               set(entry["text"].lower().split()))
                if overlap > best_score:
                    best_score = overlap
                    best_pid = pid
            if best_pid and best_score >= 3:
                entry = self._passage_lookup[best_pid]
                parsed.append({
                    "passage_id": best_pid,
                    "title": entry["title"],
                    "text": entry["text"],
                    "citation_index": idx,
                })

        return parsed

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """
        Select candidate doc_ids via keyword pre-filter, query each one,
        and return the merged result.

        Returns
        -------
        dict with keys:
            answer         (str)
            cited_passages (list of {passage_id, title, text, citation_index})
            raw_responses  (list) — full API responses for debugging
        """
        candidate_doc_ids = self._select_candidate_doc_ids(question)

        if not candidate_doc_ids:
            return {
                "answer": "",
                "cited_passages": [],
                "raw_responses": [],
            }

        messages = [{"role": "user", "content": question}]
        responses = []

        for doc_id in candidate_doc_ids:
            try:
                response = self._client.chat_completions(
                    messages=messages,
                    doc_id=doc_id,
                    enable_citations=PAGEINDEX_ENABLE_CITATIONS,
                    temperature=PAGEINDEX_TEMPERATURE,
                )
                responses.append((doc_id, response))
            except Exception:
                # Skip failed doc_ids rather than aborting the entire query
                continue

        if not responses:
            return {
                "answer": "",
                "cited_passages": [],
                "raw_responses": [],
            }

        # Merge: pick the response with the most citations; use its answer.
        # If tied, take the first.
        best_doc_id, best_response = max(
            responses,
            key=lambda x: len(
                x[1].get("choices", [{}])[0]
                    .get("message", {})
                    .get("citations", [])
            ),
        )

        message = best_response.get("choices", [{}])[0].get("message", {})
        answer = message.get("content", "").strip()
        raw_citations = message.get("citations", [])
        cited_passages = self._parse_citations(raw_citations, best_doc_id)

        # Deduplicate by passage_id, preserving citation_index order
        seen_pids: set = set()
        unique_cited = []
        for cp in cited_passages:
            if cp["passage_id"] not in seen_pids:
                seen_pids.add(cp["passage_id"])
                unique_cited.append(cp)

        return {
            "answer": answer,
            "cited_passages": unique_cited,
            "raw_responses": [r for _, r in responses],
        }
