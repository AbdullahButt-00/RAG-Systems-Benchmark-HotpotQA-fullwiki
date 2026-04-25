"""
Main evaluation script for PageIndex-RAG.

Runs the pipeline on a sample of the HotpotQA validation split
and prints a full 6-metric report.

Usage
-----
    python run_evaluation.py              # 500 samples (default)
    python run_evaluation.py --samples 100
    python run_evaluation.py --samples 7405  # full validation set
"""

import argparse
import json
import random

from tqdm import tqdm

from config import EVAL_SAMPLE_SIZE, EVAL_REPORT_PATH
from pipeline import PageIndexRAGPipeline
from evaluate import compute_all_metrics
from preprocess import _load_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PageIndex-RAG on HotpotQA")
    parser.add_argument(
        "--samples",
        type=int,
        default=EVAL_SAMPLE_SIZE,
        help=f"Number of validation examples to evaluate (default: {EVAL_SAMPLE_SIZE})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
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

    print("\nLoading PageIndex-RAG pipeline...")
    pipeline = PageIndexRAGPipeline()

    results = []
    for example in tqdm(samples, desc="Evaluating", unit="q"):
        output = pipeline.run(example["question"])
        results.append(
            {
                "question": example["question"],
                "prediction": output["answer"],
                "gold_answer": example["answer"],
                # evaluate.py expects key "retrieved_passages" — cited_passages
                # has the same shape {title, text, passage_id}
                "retrieved_passages": output["cited_passages"],
                "gold_supporting_facts": example["supporting_facts"],
                "latency_ms": output["latency_ms"],
                # Extra fields for per-question report
                "cited_titles": [p["title"] for p in output["cited_passages"]],
            }
        )

    metrics = compute_all_metrics(results)
    n = len(results)

    print("\n" + "=" * 60)
    print("PageIndex-RAG Evaluation Report — HotpotQA fullwiki")
    print("=" * 60)
    print(f"Samples Evaluated        : {n}")
    print("-" * 60)
    print(f"Exact Match (EM)         : {metrics['exact_match']:.4f}")
    print(f"Token F1                 : {metrics['token_f1']:.4f}")
    print(f"Supporting Fact F1       : {metrics['supporting_fact_f1']:.4f}")
    print(f"Recall@5                 : {metrics['recall_at_5']:.4f}")
    print(f"MRR                      : {metrics['mrr']:.4f}")
    print(f"Mean Latency (ms)        : {metrics['mean_latency_ms']:.1f}")
    print(f"P95 Latency  (ms)        : {metrics['p95_latency_ms']:.1f}")
    print("=" * 60)

    report = {
        "samples_evaluated": n,
        **metrics,
        "per_question": results,
    }
    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved → {EVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
