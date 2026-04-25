"""
Main evaluation script for Confidence-Gated Hybrid-RAG on HotpotQA fullwiki.

Usage
-----
    python run_evaluation.py                 # 500 samples (default)
    python run_evaluation.py --samples 50    # quick smoke test
    python run_evaluation.py --samples 7405  # full validation set

    # Override gate thresholds for ablation runs:
    python run_evaluation.py --conf_threshold 0.0   # always expand (ablation A2)
    python run_evaluation.py --conf_threshold 1.1   # never expand  (ablation A3)
    python run_evaluation.py --alpha 1.0            # dense-only reranking

Output
------
  Console: formatted metrics table + gate diagnostic
  File:    evaluation_report.json (metrics + per-question log)
"""

import argparse
import json
import random
from collections import Counter

from tqdm import tqdm

from config import EVAL_SAMPLE_SIZE, EVAL_REPORT_PATH, TOP_K_FINAL
from pipeline import HybridRAGPipeline
from evaluate import compute_all_metrics
from preprocess import _load_splits


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Confidence-Gated Hybrid-RAG on HotpotQA"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=EVAL_SAMPLE_SIZE,
        help=f"Number of validation examples to evaluate (default: {EVAL_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=None,
        help="Override CONF_THRESHOLD (0.0 = always expand, 1.1 = never expand)",
    )
    parser.add_argument(
        "--spread_threshold",
        type=float,
        default=None,
        help="Override SPREAD_THRESHOLD",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override RERANK_ALPHA (dense score weight in reranker)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output JSON path (default: evaluation_report.json)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_validation_split(n: int, seed: int) -> list:
    print("Loading HotpotQA validation split from local Arrow files...")
    val   = _load_splits()["validation"]
    total = len(val)
    n     = min(n, total)
    rng   = random.Random(seed)
    indices = rng.sample(range(total), n)
    samples = [val[i] for i in indices]
    print(f"Sampled {n} / {total} validation examples.")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    samples = load_validation_split(args.samples, args.seed)

    print("\nInitialising Hybrid-RAG pipeline...")
    pipeline = HybridRAGPipeline(
        conf_threshold   = args.conf_threshold,
        spread_threshold = args.spread_threshold,
        rerank_alpha     = args.alpha,
    )

    results          = []
    per_question_log = []
    # Resume from checkpoint if exists
    import os
    if os.path.exists("checkpoint.json"):
        with open("checkpoint.json") as _f:
            _ckpt = json.load(_f)
        results = _ckpt["results"]
        per_question_log = _ckpt["per_question_log"]
        done = {r["question"] for r in results}
        samples = [s for s in samples if s["question"] not in done]
        print(f"Resuming from checkpoint: {len(results)} done, {len(samples)} remaining.")

    for example in tqdm(samples, desc="Evaluating", unit="q"):
        output = pipeline.run(example["question"])
        gate   = output["gate_analysis"]

        r = {
            "question":             example["question"],
            "prediction":           output["answer"],
            "gold_answer":          example["answer"],
            "retrieved_passages":   output["retrieved_passages"],
            "gold_supporting_facts": example["supporting_facts"],
            "latency_ms":           output["latency_ms"],
            "gate_fired":           gate["gate_fired"],
        }
        results.append(r)

        per_question_log.append({
            "question":          r["question"],
            "prediction":        r["prediction"],
            "gold_answer":       r["gold_answer"],
            "gold_titles":       example["supporting_facts"]["title"],
            "retrieved_titles":  [p["title"] for p in output["retrieved_passages"]],
            "seed_titles":       [p["title"] for p in output["seed_passages"]],
            "expanded_titles":   [p["title"] for p in output["expanded_passages"]],
            "n_seeds":           output["n_seeds"],
            "n_expanded":        output["n_expanded"],
            "n_final":           output["n_final"],
            "gate_fired":        gate["gate_fired"],
            "confidence":        gate["confidence"],
            "spread":            gate["spread"],
            "trigger_reason":    gate["trigger_reason"],
            "latency_ms":        r["latency_ms"],
        })

    # ── Compute metrics ────────────────────────────────────────────────────
    metrics = compute_all_metrics(results, top_k_final=TOP_K_FINAL)
    n       = len(results)

    # ── Gate diagnostic ────────────────────────────────────────────────────
    reason_counts = Counter(q["trigger_reason"] for q in per_question_log)
    n_fired       = sum(1 for q in per_question_log if q["gate_fired"])
    avg_expanded  = (
        sum(q["n_expanded"] for q in per_question_log if q["gate_fired"]) / n_fired
        if n_fired > 0 else 0.0
    )

    # ── Print report ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Confidence-Gated Hybrid-RAG — HotpotQA Evaluation Report")
    print("=" * 64)
    print(f"Samples Evaluated       : {n}")
    print("-" * 64)
    print(f"Exact Match (EM)        : {metrics['exact_match']:.4f}")
    print(f"Token F1                : {metrics['token_f1']:.4f}")
    print(f"Supporting Fact F1      : {metrics['supporting_fact_f1']:.4f}")
    print(f"Recall@5                : {metrics['recall_at_5']:.4f}")
    print(f"Recall@{TOP_K_FINAL}               : {metrics[f'recall_at_{TOP_K_FINAL}']:.4f}")
    print(f"MRR                     : {metrics['mrr']:.4f}")
    print(f"Chain Recall            : {metrics['chain_recall']:.4f}")
    print(f"Support Coverage        : {metrics['support_coverage']:.4f}")
    print(f"Mean Latency (ms)       : {metrics['mean_latency_ms']:.1f}")
    print(f"P95 Latency  (ms)       : {metrics['p95_latency_ms']:.1f}")
    print("-" * 64)
    print(f"Gate Firing Rate        : {metrics.get('gate_firing_rate', 0.0):.4f}  "
          f"({n_fired}/{n} questions)")
    print(f"  Trigger reasons:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:<35}: {count:>4}  ({count/n*100:.1f}%)")
    if n_fired > 0:
        print(f"  Avg expanded passages  : {avg_expanded:.1f}")
    print("=" * 64)

    # ── Save report ────────────────────────────────────────────────────────
    out_path = args.output or str(EVAL_REPORT_PATH)
    report = {
        "samples_evaluated": n,
        "config": {
            "conf_threshold":   pipeline._analyzer.conf_threshold,
            "spread_threshold": pipeline._analyzer.spread_threshold,
            "rerank_alpha":     pipeline._reranker.alpha,
            "top_k_final":      TOP_K_FINAL,
        },
        "metrics":       metrics,
        "gate_summary": {
            "n_fired":         n_fired,
            "firing_rate":     n_fired / n,
            "reason_counts":   dict(reason_counts),
            "avg_expanded":    avg_expanded,
        },
        "per_question":  per_question_log,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved → {out_path}")


if __name__ == "__main__":
    main()
