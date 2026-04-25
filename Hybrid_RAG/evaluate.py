"""
Evaluation metrics for Confidence-Gated Hybrid-RAG on HotpotQA.

Metrics inherited from Graph_RAG (unchanged implementations)
-------------------------------------------------------------
  1. exact_match          — EM after answer normalization
  2. token_f1             — token-level overlap F1
  3. supporting_fact_f1   — title-level precision / recall / F1
  4. recall_at_k          — any gold title in top-k retrieved (k configurable)
  5. reciprocal_rank      — MRR over retrieved list
  6. chain_recall_at_k    — ALL gold titles in retrieved set (multi-hop success)
  7. support_coverage     — sentence-level coverage of gold evidence
  8. latency              — mean + P95

New hybrid-specific aggregate metric
-------------------------------------
  9. gate_firing_rate     — fraction of questions where graph expansion fired

Metric notes for the paper
---------------------------
  - recall_at_k is evaluated at k=TOP_K_FINAL (7) for the hybrid, and at k=5
    for all baselines. Both values are reported in compute_all_metrics().
  - gate_firing_rate is diagnostic only (not a quality metric).
  - chain_recall and support_coverage require 'sentences' in retrieved passages;
    the hybrid pipeline attaches sentences to both seeds and expanded passages.
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
# Metrics 1–2: Answer quality
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
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Metrics 3–5: Retrieval quality (title-level)
# ---------------------------------------------------------------------------

def supporting_fact_f1(retrieved_passages: list, gold_supporting_facts: dict) -> tuple:
    """Returns (precision, recall, f1) over retrieved-vs-gold title matches."""
    gold_titles = set(gold_supporting_facts["title"])
    pred_titles = {p["title"] for p in retrieved_passages}
    if not pred_titles or not gold_titles:
        return 0.0, 0.0, 0.0
    tp        = len(gold_titles & pred_titles)
    precision = tp / len(pred_titles)
    recall    = tp / len(gold_titles)
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def recall_at_k(
    retrieved_passages: list,
    gold_supporting_facts: dict,
    k: int = 5,
) -> float:
    """
    1.0 if any gold title appears in the top-k retrieved passages.

    k is configurable so this can be evaluated at both k=5 (baseline
    comparison) and k=7 (hybrid TOP_K_FINAL) in the same run.
    """
    gold_titles = set(gold_supporting_facts["title"])
    pred_titles = {p["title"] for p in retrieved_passages[:k]}
    return float(bool(gold_titles & pred_titles))


def reciprocal_rank(retrieved_passages: list, gold_supporting_facts: dict) -> float:
    """1/rank of the first passage whose title matches a gold supporting-fact title."""
    gold_titles = set(gold_supporting_facts["title"])
    for rank, passage in enumerate(retrieved_passages, start=1):
        if passage["title"] in gold_titles:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Metrics 6–7: Multi-hop specific (from Graph_RAG, unchanged)
# ---------------------------------------------------------------------------

def chain_recall_at_k(retrieved_passages: list, gold_supporting_facts: dict) -> float:
    """
    1.0 only if ALL gold supporting-fact titles appear in the retrieved set.

    This is the definitive multi-hop metric: the pipeline assembled the full
    reasoning chain. Dense-RAG typically misses one of the two required hops.
    """
    gold_titles     = set(gold_supporting_facts["title"])
    retrieved_titles = {p["title"] for p in retrieved_passages}
    return float(gold_titles.issubset(retrieved_titles))


def support_coverage(retrieved_passages: list, gold_supporting_facts: dict) -> float:
    """
    Sentence-level coverage of gold evidence.

    A (title, sent_idx) unit is covered if:
      1. The passage with that title was retrieved.
      2. The sentence at sent_idx exists in that passage's 'sentences' list.

    Returns covered_units / total_gold_units.
    Requires retrieved passages to carry a 'sentences' key (list[str]).
    The hybrid pipeline ensures seeds also have 'sentences' via corpus lookup.
    """
    gold_pairs = list(zip(
        gold_supporting_facts["title"],
        gold_supporting_facts["sent_id"],
    ))
    if not gold_pairs:
        return 0.0

    title_to_sentences: dict[str, list] = {}
    for p in retrieved_passages:
        if p["title"] not in title_to_sentences:
            title_to_sentences[p["title"]] = p.get("sentences", [])

    covered = 0
    for title, sent_idx in gold_pairs:
        if title in title_to_sentences:
            if sent_idx < len(title_to_sentences[title]):
                covered += 1

    return covered / len(gold_pairs)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_metrics(results: list, top_k_final: int = 7) -> dict:
    """
    Compute all metrics over a list of per-question result dicts.

    Each result dict must contain:
        prediction            (str)
        gold_answer           (str)
        retrieved_passages    (list[dict]) — keys: title, text, sentences, …
        gold_supporting_facts (dict)       — keys: title (list), sent_id (list)
        latency_ms            (float)
        gate_fired            (bool)       — optional; used for gate_firing_rate

    Parameters
    ----------
    results : list[dict]
    top_k_final : int
        The k value used for recall_at_k in the hybrid context.
        Recall is reported at both k=5 (baseline-compatible) and k=top_k_final.

    Returns
    -------
    dict of metric_name → float
    """
    em_scores      = []
    f1_scores      = []
    sp_f1_scores   = []
    recall5_scores = []
    recall_final_scores = []
    mrr_scores     = []
    chain_scores   = []
    coverage_scores = []
    latencies      = []
    gate_flags     = []

    for r in results:
        em_scores.append(exact_match(r["prediction"], r["gold_answer"]))
        f1_scores.append(token_f1(r["prediction"], r["gold_answer"]))

        _, _, sp_f1 = supporting_fact_f1(
            r["retrieved_passages"], r["gold_supporting_facts"]
        )
        sp_f1_scores.append(sp_f1)

        recall5_scores.append(
            recall_at_k(r["retrieved_passages"], r["gold_supporting_facts"], k=5)
        )
        recall_final_scores.append(
            recall_at_k(r["retrieved_passages"], r["gold_supporting_facts"], k=top_k_final)
        )
        mrr_scores.append(
            reciprocal_rank(r["retrieved_passages"], r["gold_supporting_facts"])
        )
        chain_scores.append(
            chain_recall_at_k(r["retrieved_passages"], r["gold_supporting_facts"])
        )
        coverage_scores.append(
            support_coverage(r["retrieved_passages"], r["gold_supporting_facts"])
        )
        latencies.append(r["latency_ms"])

        # Gate firing rate (optional field)
        gate_fired = r.get("gate_fired")
        if gate_fired is not None:
            gate_flags.append(float(gate_fired))

    metrics = {
        "exact_match":           float(np.mean(em_scores)),
        "token_f1":              float(np.mean(f1_scores)),
        "supporting_fact_f1":    float(np.mean(sp_f1_scores)),
        "recall_at_5":           float(np.mean(recall5_scores)),
        f"recall_at_{top_k_final}": float(np.mean(recall_final_scores)),
        "mrr":                   float(np.mean(mrr_scores)),
        "chain_recall":          float(np.mean(chain_scores)),
        "support_coverage":      float(np.mean(coverage_scores)),
        "mean_latency_ms":       float(np.mean(latencies)),
        "p95_latency_ms":        float(np.percentile(latencies, 95)),
    }

    if gate_flags:
        metrics["gate_firing_rate"] = float(np.mean(gate_flags))

    return metrics
