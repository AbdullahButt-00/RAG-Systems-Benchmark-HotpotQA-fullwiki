"""
Query complexity analyzer — the confidence gate for Hybrid-RAG.

This is the core novel component of the pipeline. It inspects the
dense retrieval score distribution to decide whether graph expansion
should be triggered for a given query.

Gate logic
----------
Given top-k seed passages with their dense L2-derived scores:

  confidence = max(seed_scores)
  spread     = max(seed_scores) - min(seed_scores)

  gate_fired = (confidence < CONF_THRESHOLD) OR (spread < SPREAD_THRESHOLD)

Intuition
---------
  Low confidence  → the retriever is not certain about any passage.
                    Query is likely complex (bridge or comparison type).
                    Graph expansion may surface the missing second hop.

  Low spread      → all seed scores are clustered together.
                    The retriever has no clear winner — it is equally
                    uncertain about all candidates. Expansion may help.

  High confidence + high spread → the retriever found a strong match.
                    Query is likely simple or factoid. Skip expansion.
                    This maintains precision and reduces latency.

Calibration
-----------
Before the final 500-sample evaluation, run the built-in diagnostic
function on ~50–100 validation examples to inspect the score distribution
and verify CONF_THRESHOLD / SPREAD_THRESHOLD make sense for your index.

    from query_analyzer import run_threshold_diagnostic
    run_threshold_diagnostic(pipeline.dense, samples)
"""

from config import CONF_THRESHOLD, SPREAD_THRESHOLD


class QueryAnalyzer:
    """
    Confidence-based gate that decides when to trigger graph expansion.

    Parameters
    ----------
    conf_threshold : float
        If max(seed_scores) < this, graph expansion is triggered.
        Default: CONF_THRESHOLD from config.py (0.40)
    spread_threshold : float
        If score spread < this, graph expansion is triggered.
        Default: SPREAD_THRESHOLD from config.py (0.05)
    """

    def __init__(
        self,
        conf_threshold: float  = CONF_THRESHOLD,
        spread_threshold: float = SPREAD_THRESHOLD,
    ) -> None:
        self.conf_threshold   = conf_threshold
        self.spread_threshold = spread_threshold

    def analyze(self, seed_passages: list[dict]) -> dict:
        """
        Analyze seed passage scores and decide whether to trigger graph expansion.

        Parameters
        ----------
        seed_passages : list[dict]
            Output of DenseRetriever.retrieve_seed(). Each dict must have a
            'score' key (float). Order does not matter.

        Returns
        -------
        dict with:
            gate_fired     (bool)   — True  = trigger graph expansion
                                      False = skip, use seeds only
            is_complex     (bool)   — alias for gate_fired (explicit naming)
            confidence     (float)  — max dense score across seeds
            spread         (float)  — max - min dense score
            trigger_reason (str)    — human-readable gate decision label:
                                      'low_confidence'
                                      'low_spread'
                                      'low_confidence_and_spread'
                                      'no_trigger'
                                      'no_seeds'
        """
        scores = [
            float(p["score"])
            for p in seed_passages
            if p.get("score") is not None
        ]

        if not scores:
            return {
                "gate_fired":     True,
                "is_complex":     True,
                "confidence":     0.0,
                "spread":         0.0,
                "trigger_reason": "no_seeds",
            }

        confidence = max(scores)
        spread     = max(scores) - min(scores) if len(scores) > 1 else 0.0

        low_confidence = confidence < self.conf_threshold
        low_spread     = spread     < self.spread_threshold
        gate_fired     = low_confidence or low_spread

        if low_confidence and low_spread:
            reason = "low_confidence_and_spread"
        elif low_confidence:
            reason = "low_confidence"
        elif low_spread:
            reason = "low_spread"
        else:
            reason = "no_trigger"

        return {
            "gate_fired":     gate_fired,
            "is_complex":     gate_fired,
            "confidence":     confidence,
            "spread":         spread,
            "trigger_reason": reason,
        }


# ---------------------------------------------------------------------------
# Calibration diagnostic
# ---------------------------------------------------------------------------

def run_threshold_diagnostic(
    dense_retriever,
    samples: list[dict],
    top_k: int = 5,
) -> None:
    """
    Print score distribution statistics to help calibrate gate thresholds.

    Run this on 50–100 HotpotQA validation examples BEFORE the main
    evaluation to understand where confidence and spread values fall,
    and whether CONF_THRESHOLD / SPREAD_THRESHOLD need adjustment.

    IMPORTANT — score polarity
    --------------------------
    LlamaIndex's FaissVectorStore with IndexFlatL2 returns scores as
    NEGATIVE L2 distances: score = -dist, so scores are in (-∞, 0].
    For unit-normalized embeddings (multi-qa-MiniLM-L6-cos-v1) the
    practical range is roughly [-1.41, 0].

    If scores are negative the default CONF_THRESHOLD = 0.40 will ALWAYS
    fire (since max(scores) < 0 < 0.40), making the gate equivalent to
    "always expand". To fix this, set CONF_THRESHOLD to a NEGATIVE value
    that separates confident from uncertain retrieval in the actual range.
    Example: CONF_THRESHOLD = -0.50 means "expand when the best score
    is worse than -0.50" (i.e., cosine similarity below ~0.88).

    This function prints the score distribution so you can choose the
    right threshold for your index.

    Usage
    -----
        from query_analyzer import run_threshold_diagnostic
        from retriever import DenseRetriever

        dense = DenseRetriever()
        dense.load_index()

        # samples is a list of HotpotQA example dicts
        run_threshold_diagnostic(dense, samples[:100])

    Parameters
    ----------
    dense_retriever : DenseRetriever
        Loaded retriever.
    samples : list[dict]
        HotpotQA examples with a 'question' key.
    top_k : int
        Number of seeds to retrieve per question.
    """
    import statistics

    confidences = []
    spreads     = []
    reasons     = {}
    analyzer    = QueryAnalyzer()

    print(f"Running threshold diagnostic on {len(samples)} questions...")

    # Single pass: retrieve seeds, analyze, collect ALL stats from the real run.
    # Do NOT re-run analyze() with partial data — that would give wrong reasons.
    for ex in samples:
        seeds  = dense_retriever.retrieve_seed(ex["question"], top_k=top_k)
        result = analyzer.analyze(seeds)
        confidences.append(result["confidence"])
        spreads.append(result["spread"])
        r = result["trigger_reason"]
        reasons[r] = reasons.get(r, 0) + 1

    n = len(confidences)

    # ── Score polarity warning ─────────────────────────────────────────────
    if max(confidences) <= 0:
        print("\n  ⚠  WARNING: All confidence scores are ≤ 0.")
        print("     This indicates LlamaIndex is returning NEGATIVE L2 distances.")
        print("     CONF_THRESHOLD = 0.40 will ALWAYS fire with these scores.")
        print(f"    Suggested CONF_THRESHOLD: around {statistics.median(confidences):.3f}")
        print("     Update CONF_THRESHOLD in config.py to a value in the range above.")
    elif min(confidences) < 0:
        print("\n  ⚠  WARNING: Some confidence scores are negative.")
        print("     Check whether CONF_THRESHOLD is in the right range for your index.")

    print("\n── Confidence (max seed score) ──────────────────────────")
    print(f"  min    : {min(confidences):.4f}")
    print(f"  max    : {max(confidences):.4f}")
    print(f"  mean   : {statistics.mean(confidences):.4f}")
    print(f"  median : {statistics.median(confidences):.4f}")
    if n > 1:
        print(f"  stdev  : {statistics.stdev(confidences):.4f}")

    # Adaptive percentile report — shows the three quartiles of the distribution
    lo  = statistics.median(confidences) - statistics.stdev(confidences) if n > 1 else confidences[0] - 0.1
    mid = statistics.median(confidences)
    hi  = statistics.median(confidences) + statistics.stdev(confidences) if n > 1 else confidences[0] + 0.1
    print(f"\n  % below {lo:.2f} : {sum(c < lo  for c in confidences) / n * 100:.1f}%")
    print(f"  % below {mid:.2f} : {sum(c < mid for c in confidences) / n * 100:.1f}%")
    print(f"  % below {hi:.2f} : {sum(c < hi  for c in confidences) / n * 100:.1f}%")
    print(f"\n  current CONF_THRESHOLD = {analyzer.conf_threshold}")
    print(f"  % that would fire conf gate : "
          f"{sum(c < analyzer.conf_threshold for c in confidences) / n * 100:.1f}%")

    print("\n── Spread (max - min seed score) ────────────────────────")
    print(f"  min    : {min(spreads):.4f}")
    print(f"  max    : {max(spreads):.4f}")
    print(f"  mean   : {statistics.mean(spreads):.4f}")
    print(f"  median : {statistics.median(spreads):.4f}")
    print(f"\n  % below 0.05 : {sum(s < 0.05 for s in spreads) / n * 100:.1f}%")
    print(f"  % below 0.10 : {sum(s < 0.10 for s in spreads) / n * 100:.1f}%")
    print(f"\n  current SPREAD_THRESHOLD = {analyzer.spread_threshold}")
    print(f"  % that would fire spread gate : "
          f"{sum(s < analyzer.spread_threshold for s in spreads) / n * 100:.1f}%")

    print(f"\n── Gate firing rate at current thresholds ───────────────")
    fired = sum(
        1 for c, s in zip(confidences, spreads)
        if c < analyzer.conf_threshold or s < analyzer.spread_threshold
    )
    print(f"  CONF_THRESHOLD   = {analyzer.conf_threshold}")
    print(f"  SPREAD_THRESHOLD = {analyzer.spread_threshold}")
    print(f"  Gate fires on    : {fired}/{n} ({fired/n*100:.1f}%)")

    print(f"\n── Trigger reason breakdown (from real {top_k}-seed runs) ──")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:<35}: {count:>4}  ({count/n*100:.1f}%)")
    print()
