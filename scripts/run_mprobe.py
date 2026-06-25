"""
FormInv M-probe: Contamination Detection via Named vs. Anonymous Queries.

M(model) = accuracy(named) - accuracy(NL_anonymous)

A positive M means the model benefits from the Lean4 theorem NAME (potential
memorisation from training data), not just from reasoning about the math.

Usage:
  cd /Users/noel.thomas/chaos-logic-bench/extensions/forminv
  python scripts/run_mprobe.py
  python scripts/run_mprobe.py --no-cache
  python scripts/run_mprobe.py --models gpt-4o,claude-sonnet-4-6

All theorems have ground_truth=TRUE, so accuracy = fraction answering TRUE.
Parse failures (parsed=None) are excluded from the accuracy denominator but
are reported as N_parsed/N_total so the M score remains interpretable.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from forminv.eval.providers import call_model

# -- Model registry ------------------------------------------------------------
ALL_MODELS = [
    ("openai", "gpt-4o"),
    ("openai", "gpt-4o-mini"),
    ("openai", "o4-mini"),
    ("anthropic", "claude-sonnet-4-6"),
    ("anthropic", "claude-haiku-4-5"),
    ("deepseek", "deepseek-chat"),
    ("deepseek", "deepseek-reasoner"),
    ("gemini", "gemini-2.5-flash"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
]

DATASET_PATH = Path("data/generated/formInv_mprobe_50.jsonl")
RAW_LOG_PATH = Path("artifacts/mprobe_raw.jsonl")
RESULTS_PATH = Path("artifacts/mprobe_results.json")
REPORT_PATH = Path("artifacts/mprobe_report.md")


def load_probe_items() -> list[dict]:
    items = []
    with open(DATASET_PATH) as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items


def run_condition(
    items: list[dict], provider: str, model: str, condition: str, use_cache: bool, raw_log
) -> list[dict | None]:
    """Run one model on one condition ('named' or 'anonymous').

    Returns list of parsed labels (True/False/None) per item.
    """
    assert condition in ("named", "anonymous")
    results = []
    for i, item in enumerate(items):
        question = item["named_query"] if condition == "named" else item["anonymous_query"]
        try:
            resp = call_model(question, provider=provider, model=model, use_cache=use_cache)
            parsed = resp.parsed  # "TRUE", "FALSE", or None
        except Exception as e:
            print(f"    [ERR] {provider}/{model} {condition} item {i}: {e}")
            parsed = None
            resp_raw = f"ERROR: {e}"
            resp_latency = 0.0
        else:
            resp_raw = resp.raw
            resp_latency = resp.latency_s

        # Append-write to raw log immediately (crash-safe)
        raw_log.write(
            json.dumps(
                {
                    "model": model,
                    "provider": provider,
                    "condition": condition,
                    "theorem_id": item["theorem_id"],
                    "mathlib_name": item["mathlib_name"],
                    "question": question,
                    "raw": resp_raw,
                    "parsed": parsed,
                    "latency_s": resp_latency,
                }
            )
            + "\n"
        )
        raw_log.flush()

        results.append(parsed)

    return results


def compute_m_score(named_results: list[str | None], anon_results: list[str | None]) -> dict:
    """Compute accuracy for each condition and M = acc_named - acc_anon.

    All ground truths are TRUE -> accuracy = fraction parsing to TRUE.
    Parse failures are excluded from the denominator.
    """

    def _acc(results):
        valid = [r for r in results if r is not None]
        n_true = sum(1 for r in valid if r == "TRUE")
        n_total = len(results)
        n_valid = len(valid)
        acc = n_true / n_valid if n_valid else 0.0
        return acc, n_valid, n_total

    acc_named, n_valid_named, n_total = _acc(named_results)
    acc_anon, n_valid_anon, _ = _acc(anon_results)
    m_score = acc_named - acc_anon
    return {
        "acc_named": acc_named,
        "acc_anon": acc_anon,
        "m_score": m_score,
        "n_valid_named": n_valid_named,
        "n_valid_anon": n_valid_anon,
        "n_total": n_total,
    }


def interpret_m(m: float) -> str:
    if m > 0.12:
        return "Strong contamination signal"
    elif m > 0.05:
        return "Moderate contamination signal"
    elif m >= -0.03:
        return "Minimal / no contamination"
    else:
        return "Names hurt (unfamiliar Lean4 names)"


def write_report(all_results: dict) -> str:
    lines = [
        "# FormInv M-probe Results",
        "",
        "M(model) = Acc_named - Acc_NL_anonymous",
        "",
        "All 50 theorems have ground_truth=TRUE.  Accuracy = fraction",
        "answering TRUE (parse failures excluded from denominator).",
        "",
        "| Model | Acc_named | Acc_NL | M score | N_named | N_NL | Interpretation |",
        "|-------|-----------|--------|---------|---------|------|----------------|",
    ]
    for model_key, r in sorted(all_results.items(), key=lambda x: -x[1]["m_score"]):
        lines.append(
            f"| {model_key} "
            f"| {r['acc_named'] * 100:.1f}% "
            f"| {r['acc_anon'] * 100:.1f}% "
            f"| {r['m_score'] * 100:+.1f}% "
            f"| {r['n_valid_named']}/{r['n_total']} "
            f"| {r['n_valid_anon']}/{r['n_total']} "
            f"| {r['interpretation']} |"
        )

    lines += [
        "",
        "## Methodology",
        "",
        "- **Named query**: `Is the Lean4 Mathlib4 theorem {mathlib_name} true?`",
        "- **Anonymous query**: `{canonical_nl}` (clean NL statement)",
        "- Both wrapped by providers.USER_TEMPLATE (adds `Answer TRUE or FALSE.`)",
        "- Parse failures excluded from accuracy denominator.",
        "- Dataset: `data/generated/formInv_mprobe_50.jsonl` (50 theorems, all TRUE)",
        "- Reference: Thomas 2026, arXiv:2605.04895 (ChaosBench-Logic M-probe methodology)",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="FormInv M-probe contamination detector")
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    parser.add_argument("--no-cache", action="store_true", help="Disable response cache (re-runs all API calls)")
    args = parser.parse_args()

    use_cache = not args.no_cache

    # Select models
    models_to_run = ALL_MODELS
    if args.models != "all":
        selected = set(args.models.split(","))
        models_to_run = [(p, m) for p, m in ALL_MODELS if m in selected]
        if not models_to_run:
            print(f"[ERROR] No models matched: {selected}")
            sys.exit(1)

    items = load_probe_items()
    print(f"Loaded {len(items)} probe pairs from {DATASET_PATH}")

    RAW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Use append mode so partial re-runs (e.g. single provider) don't clobber
    # earlier results. The results JSON is always fully recomputed from scratch.
    with open(RAW_LOG_PATH, "a") as raw_log:
        for provider, model in models_to_run:
            model_key = f"{provider}/{model}"
            print(f"\n=== {model_key} ===")

            print(f"  [named] running {len(items)} items...")
            named_results = run_condition(items, provider, model, "named", use_cache, raw_log)

            print(f"  [anonymous] running {len(items)} items...")
            anon_results = run_condition(items, provider, model, "anonymous", use_cache, raw_log)

            scores = compute_m_score(named_results, anon_results)
            scores["interpretation"] = interpret_m(scores["m_score"])
            all_results[model_key] = scores

            print(
                f"  Acc_named={scores['acc_named'] * 100:.1f}%  "
                f"Acc_NL={scores['acc_anon'] * 100:.1f}%  "
                f"M={scores['m_score'] * 100:+.1f}%  "
                f"-> {scores['interpretation']}"
            )

    # Save JSON results
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to: {RESULTS_PATH}")

    # Save markdown report
    report_text = write_report(all_results)
    REPORT_PATH.write_text(report_text)
    print(f"Report saved to:  {REPORT_PATH}")

    # Print ASCII table
    print("\n" + "=" * 80)
    print("M-PROBE SUMMARY")
    print("=" * 80)
    print(f"{'Model':<45} {'Acc_named':>10} {'Acc_NL':>8} {'M score':>9} {'Interp'}")
    print("-" * 80)
    for model_key, r in sorted(all_results.items(), key=lambda x: -x[1]["m_score"]):
        print(
            f"{model_key:<45} "
            f"{r['acc_named'] * 100:>9.1f}% "
            f"{r['acc_anon'] * 100:>7.1f}% "
            f"{r['m_score'] * 100:>+8.1f}% "
            f"  {r['interpretation']}"
        )


if __name__ == "__main__":
    main()
