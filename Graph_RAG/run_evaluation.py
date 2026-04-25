"""
Main evaluation script for Graph-RAG on HotpotQA fullwiki.

Usage
-----
    python run_evaluation.py                 # 500 samples (default)
    python run_evaluation.py --samples 100   # quick smoke test
    python run_evaluation.py --samples 7405  # full validation set
"""

import argparse
import json
import random

from tqdm import tqdm

from config import EVAL_SAMPLE_SIZE, EVAL_REPORT_PATH
from pipeline import GraphRAGPipeline
from evaluate import compute_all_metrics
from preprocess import _load_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Graph-RAG on HotpotQA")
    parser.add_argument(
        "--samples",
        type=int,
        default=EVAL_SAMPLE_SIZE,
        help=f"Number of validation examples to evaluate (default: {EVAL_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )
    return parser.parse_args()


def load_validation_split(n: int, seed: int):
    print("Loading HotpotQA validation split from local Arrow files...")
    val = _load_splits()["validation"]
    total = len(val)
    n = min(n, total)
    rng = random.Random(seed)
    indices = rng.sample(range(total), n)
    samples = [val[i] for i in indices]
    print(f"Sampled {n} / {total} validation examples.")
    return samples


def main() -> None:
    args = parse_args()
    samples = load_validation_split(args.samples, args.seed)

    print("\nLoading Graph-RAG pipeline (FAISS + graph + corpus)...")
    pipeline = GraphRAGPipeline()

    results = []
    per_question_log = []

    for example in tqdm(samples, desc="Evaluating", unit="q"):
        output = pipeline.run(example["question"])

        r = {
            "question": example["question"],
            "prediction": output["answer"],
            "gold_answer": example["answer"],
            "retrieved_passages": output["retrieved_passages"],
            "gold_supporting_facts": example["supporting_facts"],
            "latency_ms": output["latency_ms"],
        }
        results.append(r)

        per_question_log.append({
            "question": r["question"],
            "prediction": r["prediction"],
            "gold_answer": r["gold_answer"],
            "n_retrieved": len(output["retrieved_passages"]),
            "n_seeds": len(output["seed_passages"]),
            "n_expanded": len(output["expanded_passages"]),
            "retrieved_titles": [p["title"] for p in output["retrieved_passages"]],
            "gold_titles": example["supporting_facts"]["title"],
            "latency_ms": r["latency_ms"],
        })

    metrics = compute_all_metrics(results)
    n = len(results)

    print("\n" + "=" * 60)
    print("Graph-RAG Evaluation Report — HotpotQA fullwiki")
    print("=" * 60)
    print(f"Samples Evaluated      : {n}")
    print("-" * 60)
    print(f"Exact Match (EM)       : {metrics['exact_match']:.4f}")
    print(f"Token F1               : {metrics['token_f1']:.4f}")
    print(f"Supporting Fact F1     : {metrics['supporting_fact_f1']:.4f}")
    print(f"Recall@5               : {metrics['recall_at_5']:.4f}")
    print(f"MRR                    : {metrics['mrr']:.4f}")
    print(f"Chain Recall@10        : {metrics['chain_recall_at_10']:.4f}")
    print(f"Support Coverage       : {metrics['support_coverage']:.4f}")
    print(f"Mean Latency (ms)      : {metrics['mean_latency_ms']:.1f}")
    print(f"P95 Latency  (ms)      : {metrics['p95_latency_ms']:.1f}")
    print("=" * 60)

    report = {
        "samples_evaluated": n,
        "metrics": metrics,
        "per_question": per_question_log,
    }
    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved → {EVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
