"""
Interactive demo: run the hybrid pipeline on individual questions.

Usage
-----
    python demo.py
    python demo.py --question "Who was the director of Inception?"
    python demo.py --verbose    # print gate analysis and passage scores
    python demo.py --diagnostic # print threshold diagnostic on 50 val examples

The demo visualises the gate decision, which passages were retrieved from
which stage, and the final reranked context sent to the LLM.
"""

import argparse

from config import TOP_K_FINAL, CONF_THRESHOLD, SPREAD_THRESHOLD


EXAMPLE_QUESTIONS = [
    "What nationality is the director of the film Forrest Gump?",
    "What is the capital of the country where the Eiffel Tower is located?",
    "Who was the first president of the country that hosted the 1936 Summer Olympics?",
    "What award did the actress who played Katniss Everdeen win for Silver Linings Playbook?",
    "In what city was the person born who invented the telephone?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid-RAG interactive demo")
    parser.add_argument(
        "--question", type=str, default=None,
        help="Single question to answer. If omitted, runs example questions."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print gate analysis, passage scores, and source labels."
    )
    parser.add_argument(
        "--diagnostic", action="store_true",
        help="Run threshold diagnostic on 50 validation examples, then exit."
    )
    return parser.parse_args()


def print_result(question: str, output: dict, verbose: bool = False) -> None:
    gate = output["gate_analysis"]

    print(f"\n{'─'*60}")
    print(f"Q: {question}")
    print(f"A: {output['answer']}")
    print(f"   Gate fired : {gate['gate_fired']}  "
          f"(conf={gate['confidence']:.3f}, spread={gate['spread']:.3f}, "
          f"reason={gate['trigger_reason']})")
    print(f"   Seeds: {output['n_seeds']}  |  Expanded: {output['n_expanded']}  "
          f"|  Final: {output['n_final']}  |  Latency: {output['latency_ms']:.0f} ms")

    if verbose:
        print(f"\n  Final context sent to LLM (top-{TOP_K_FINAL}):")
        for i, p in enumerate(output["retrieved_passages"], 1):
            score_str = f"{p.get('final_score', 0.0):.3f}"
            src_label = f"[{p.get('source', '?'):>14}]"
            print(f"  [{i}] {src_label} score={score_str}  «{p['title']}»")
            print(f"       {p['text'][:120]}...")

    print()


def main() -> None:
    args = parse_args()

    print("Loading Hybrid-RAG pipeline...")
    from pipeline import HybridRAGPipeline
    pipeline = HybridRAGPipeline()

    # ── Threshold diagnostic mode ──────────────────────────────────────────
    if args.diagnostic:
        print("\nRunning threshold diagnostic on 50 validation examples...")
        from preprocess import _load_splits
        import random
        val = _load_splits()["validation"]
        rng = random.Random(42)
        samples = [val[i] for i in rng.sample(range(len(val)), 50)]
        from query_analyzer import run_threshold_diagnostic
        run_threshold_diagnostic(pipeline._dense, samples)
        return

    # ── Single question mode ───────────────────────────────────────────────
    if args.question:
        output = pipeline.run(args.question)
        print_result(args.question, output, verbose=args.verbose)
        return

    # ── Example questions mode ─────────────────────────────────────────────
    print(f"\nRunning {len(EXAMPLE_QUESTIONS)} example questions...\n")
    print(f"Config: CONF_THRESHOLD={CONF_THRESHOLD}, "
          f"SPREAD_THRESHOLD={SPREAD_THRESHOLD}, "
          f"TOP_K_FINAL={TOP_K_FINAL}")

    for q in EXAMPLE_QUESTIONS:
        output = pipeline.run(q)
        print_result(q, output, verbose=args.verbose)


if __name__ == "__main__":
    main()
