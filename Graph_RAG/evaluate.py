"""
All 8 evaluation metrics for Graph-RAG on HotpotQA.

Metrics 1-6 (unchanged from Dense-RAG):
    exact_match, token_f1, supporting_fact_f1, recall_at_k,
    reciprocal_rank, latency (mean + p95)

Metrics 7-8 (new Graph-RAG metrics):
    chain_recall_at_k  — did graph expansion collect the FULL multi-hop chain?
    support_coverage   — sentence-level coverage of gold evidence
"""

import re
import string
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Metrics 1-2: Answer quality
# ---------------------------------------------------------------------------

def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Metrics 3-5: Retrieval quality (title-level)
# ---------------------------------------------------------------------------

def supporting_fact_f1(retrieved_passages: list, gold_supporting_facts: dict) -> tuple:
    """Returns (precision, recall, f1) over title matches."""
    gold_titles = set(gold_supporting_facts["title"])
    pred_titles = {p["title"] for p in retrieved_passages}
    if not pred_titles or not gold_titles:
        return 0.0, 0.0, 0.0
    tp = len(gold_titles & pred_titles)
    precision = tp / len(pred_titles)
    recall = tp / len(gold_titles)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def recall_at_k(retrieved_passages: list, gold_supporting_facts: dict) -> float:
    """1.0 if any gold title appears in the top-5 retrieved passages."""
    gold_titles = set(gold_supporting_facts["title"])
    pred_titles = {p["title"] for p in retrieved_passages[:5]}
    return float(bool(gold_titles & pred_titles))


def reciprocal_rank(retrieved_passages: list, gold_supporting_facts: dict) -> float:
    """1/rank of the first retrieved passage matching a gold supporting-fact title."""
    gold_titles = set(gold_supporting_facts["title"])
    for rank, passage in enumerate(retrieved_passages, start=1):
        if passage["title"] in gold_titles:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Metric 7: Chain Recall@k  (Graph-RAG specific)
# ---------------------------------------------------------------------------

def chain_recall_at_k(retrieved_passages: list, gold_supporting_facts: dict) -> float:
    """
    1.0 only if ALL gold supporting-fact titles appear somewhere in the
    retrieved passage list.

    This is the defining Graph-RAG metric: measures whether the full
    multi-hop reasoning chain (both/all supporting documents) was assembled
    by the graph expansion. Dense-RAG often misses one of the two hops.
    """
    gold_titles = set(gold_supporting_facts["title"])
    retrieved_titles = {p["title"] for p in retrieved_passages}
    return float(gold_titles.issubset(retrieved_titles))


# ---------------------------------------------------------------------------
# Metric 8: Support Coverage  (Graph-RAG specific)
# ---------------------------------------------------------------------------

def support_coverage(retrieved_passages: list, gold_supporting_facts: dict) -> float:
    """
    Fine-grained sentence-level coverage of gold evidence.

    For each gold (title, sent_idx) pair a unit is "covered" if:
        1. The passage with that title was retrieved, AND
        2. The sentence at sent_idx exists in that passage's sentences list.

    Returns covered_units / total_gold_units.

    Requires retrieved passages to carry a `sentences` key (list of strings),
    which is populated from corpus.json via graph_retriever.py.
    """
    gold_pairs = list(zip(
        gold_supporting_facts["title"],
        gold_supporting_facts["sent_id"],
    ))
    if not gold_pairs:
        return 0.0

    # Build lookup: retrieved title → sentences list
    title_to_sentences: dict[str, list] = {}
    for p in retrieved_passages:
        if p["title"] not in title_to_sentences:
            title_to_sentences[p["title"]] = p.get("sentences", [])

    covered = 0
    for title, sent_idx in gold_pairs:
        if title in title_to_sentences:
            sents = title_to_sentences[title]
            if sent_idx < len(sents):
                covered += 1

    return covered / len(gold_pairs)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_metrics(results: list) -> dict:
    """
    Compute all 8 metrics over a list of per-question result dicts.

    Each result dict must contain:
        prediction            (str)
        gold_answer           (str)
        retrieved_passages    (list[dict])  — keys: title, text, sentences, …
        gold_supporting_facts (dict)        — keys: title (list), sent_id (list)
        latency_ms            (float)

    Returns
    -------
    dict of metric_name → float
    """
    em_scores, f1_scores, sp_f1_scores = [], [], []
    recall5_scores, mrr_scores = [], []
    chain_recall_scores, coverage_scores = [], []
    latencies = []

    for r in results:
        em_scores.append(exact_match(r["prediction"], r["gold_answer"]))
        f1_scores.append(token_f1(r["prediction"], r["gold_answer"]))
        _, _, sp_f1 = supporting_fact_f1(
            r["retrieved_passages"], r["gold_supporting_facts"]
        )
        sp_f1_scores.append(sp_f1)
        recall5_scores.append(
            recall_at_k(r["retrieved_passages"], r["gold_supporting_facts"])
        )
        mrr_scores.append(
            reciprocal_rank(r["retrieved_passages"], r["gold_supporting_facts"])
        )
        chain_recall_scores.append(
            chain_recall_at_k(r["retrieved_passages"], r["gold_supporting_facts"])
        )
        coverage_scores.append(
            support_coverage(r["retrieved_passages"], r["gold_supporting_facts"])
        )
        latencies.append(r["latency_ms"])

    return {
        "exact_match": float(np.mean(em_scores)),
        "token_f1": float(np.mean(f1_scores)),
        "supporting_fact_f1": float(np.mean(sp_f1_scores)),
        "recall_at_5": float(np.mean(recall5_scores)),
        "mrr": float(np.mean(mrr_scores)),
        "chain_recall_at_10": float(np.mean(chain_recall_scores)),
        "support_coverage": float(np.mean(coverage_scores)),
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }
