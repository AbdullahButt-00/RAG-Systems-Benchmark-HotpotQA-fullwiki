import re
import string
from collections import Counter

import numpy as np


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


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


def supporting_fact_f1(retrieved_passages: list, gold_supporting_facts: dict) -> tuple:
    """Returns (precision, recall, f1) comparing retrieved titles to gold titles."""
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
    """1.0 if any gold supporting fact title appears in retrieved passages, else 0.0."""
    gold_titles = set(gold_supporting_facts["title"])
    pred_titles = {p["title"] for p in retrieved_passages}
    return float(bool(gold_titles & pred_titles))


def reciprocal_rank(retrieved_passages: list, gold_supporting_facts: dict) -> float:
    """1/rank of the first retrieved passage whose title matches a gold title."""
    gold_titles = set(gold_supporting_facts["title"])
    for rank, passage in enumerate(retrieved_passages, start=1):
        if passage["title"] in gold_titles:
            return 1.0 / rank
    return 0.0


def compute_all_metrics(results: list) -> dict:
    """
    Args:
        results: list of dicts, each with:
            - prediction (str)
            - gold_answer (str)
            - retrieved_passages (list of dicts with 'title', 'text', etc.)
            - gold_supporting_facts (dict with 'title' and 'sent_id' keys)
            - latency_ms (float)
    Returns:
        dict of metric name → value
    """
    em_scores, f1_scores, sp_f1_scores = [], [], []
    recall5_scores, mrr_scores, latencies = [], [], []

    for r in results:
        em_scores.append(exact_match(r["prediction"], r["gold_answer"]))
        f1_scores.append(token_f1(r["prediction"], r["gold_answer"]))
        _, _, sp_f1 = supporting_fact_f1(r["retrieved_passages"], r["gold_supporting_facts"])
        sp_f1_scores.append(sp_f1)
        recall5_scores.append(recall_at_k(r["retrieved_passages"], r["gold_supporting_facts"]))
        mrr_scores.append(reciprocal_rank(r["retrieved_passages"], r["gold_supporting_facts"]))
        latencies.append(r["latency_ms"])

    return {
        "exact_match": float(np.mean(em_scores)),
        "token_f1": float(np.mean(f1_scores)),
        "supporting_fact_f1": float(np.mean(sp_f1_scores)),
        "recall_at_5": float(np.mean(recall5_scores)),
        "mrr": float(np.mean(mrr_scores)),
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }
