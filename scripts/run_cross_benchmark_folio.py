"""
Cross-benchmark FOLIO/MMLU-formal_logic evaluation for FormInv.

Runs all 9 FormInv models on 60 TRUE/FALSE formal-logic items from
MMLU formal_logic (FOLIO was gated and unavailable).

Usage:
    cd /Users/noel.thomas/forminv
    .venv/bin/python scripts/run_cross_benchmark_folio.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from forminv.eval.providers import call_model

# ---------------------------------------------------------------------------
# Item loading
# ---------------------------------------------------------------------------
ITEMS_PATH = Path("data/raw/mmlu_formal_logic_60.json")

PROMPT_TEMPLATE = """Formal logic question:
{question}

Proposed answer: \"{presented_answer}\"

Is this the correct answer to the formal logic question above?
Answer TRUE if it is correct, FALSE if it is not correct."""

# ---------------------------------------------------------------------------
# Model roster  (provider, model_id, ig_key)
# ig_key must match IG_SCORES keys exactly
# ---------------------------------------------------------------------------
MODELS = [
    ("openai", "gpt-4o-mini", "gpt-4o-mini"),
    ("openai", "gpt-4o", "gpt-4o"),
    ("openai", "o4-mini", "o4-mini"),
    ("anthropic", "claude-haiku-4-5", "claude-haiku-4-5"),
    ("anthropic", "claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("deepseek", "deepseek-chat", "deepseek-chat"),
    ("deepseek", "deepseek-reasoner", "deepseek-reasoner"),
    ("gemini", "gemini-2.5-flash", "gemini-2.5-flash"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct", "llama-3.3-70b-instruct"),
]

IG_SCORES = {
    "deepseek-chat": 6.9,
    "gpt-4o": 7.0,
    "o4-mini": 7.6,
    "gemini-2.5-flash": 8.4,
    "gpt-4o-mini": 8.7,
    "deepseek-reasoner": 8.8,
    "claude-sonnet-4-6": 10.7,
    "llama-3.3-70b-instruct": 13.5,
    "claude-haiku-4-5": 19.4,
}

# FormInv main-task accuracy (for partial correlation)
FORMINV_ACC = {
    "deepseek-chat": 96.4,
    "gpt-4o": 94.2,
    "o4-mini": 96.4,
    "gemini-2.5-flash": 95.0,
    "gpt-4o-mini": 94.3,
    "deepseek-reasoner": 93.0,
    "claude-sonnet-4-6": 93.2,
    "llama-3.3-70b-instruct": 90.5,
    "claude-haiku-4-5": 86.0,
}

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------


def run_all():
    with open(ITEMS_PATH) as f:
        items = json.load(f)

    results = {}

    for provider, model_id, ig_key in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Model: {provider}/{model_id}  (ig_key={ig_key})")
        print(f"{'=' * 60}")

        per_item = []
        n_correct = 0
        n_invalid = 0

        for i, item in enumerate(items):
            question = PROMPT_TEMPLATE.format(
                question=item["question"].strip(),
                presented_answer=item["presented_answer"],
            )
            try:
                resp = call_model(question, provider, model_id, use_cache=True)
            except Exception as e:
                print(f"  [{i + 1:02d}/{len(items)}] ERROR: {e}")
                per_item.append(
                    {
                        "item_id": item["item_id"],
                        "ground_truth": item["ground_truth"],
                        "response_raw": None,
                        "parsed": None,
                        "correct": False,
                        "error": str(e),
                    }
                )
                n_invalid += 1
                continue

            correct = resp.parsed == item["ground_truth"]
            if resp.parsed is None:
                n_invalid += 1

            per_item.append(
                {
                    "item_id": item["item_id"],
                    "ground_truth": item["ground_truth"],
                    "response_raw": resp.raw[:200],
                    "parsed": resp.parsed,
                    "correct": correct,
                    "latency_s": round(resp.latency_s, 3),
                }
            )
            if correct:
                n_correct += 1

            if (i + 1) % 10 == 0:
                print(f"  [{i + 1:02d}/{len(items)}] running acc={n_correct}/{i + 1} invalid={n_invalid}")

        acc = n_correct / len(items)
        print(f"  FINAL: acc={acc:.3f}  n_correct={n_correct}/{len(items)}  invalid={n_invalid}")

        results[ig_key] = {
            "provider": provider,
            "model_id": model_id,
            "accuracy": acc,
            "n_correct": n_correct,
            "n_invalid": n_invalid,
            "n_total": len(items),
            "per_item": per_item,
        }

    return results


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def compute_correlations(results):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import LinearRegression

    model_keys = list(results.keys())
    # Only include models present in IG_SCORES
    model_keys = [k for k in model_keys if k in IG_SCORES]

    igs = np.array([IG_SCORES[k] for k in model_keys])
    bench_acc = np.array([results[k]["accuracy"] for k in model_keys])
    forminv_acc = np.array([FORMINV_ACC[k] / 100.0 for k in model_keys])

    bench_err = 1.0 - bench_acc

    # Raw correlations
    r_pearson, p_pearson = pearsonr(igs, bench_err)
    r_spearman, p_spearman = spearmanr(igs, bench_err)

    # Partial correlation: IG vs bench_err controlling for forminv_acc
    X_acc = forminv_acc.reshape(-1, 1)

    reg_ig = LinearRegression().fit(X_acc, igs)
    ig_resid = igs - reg_ig.predict(X_acc)

    reg_bench = LinearRegression().fit(X_acc, bench_err)
    bench_resid = bench_err - reg_bench.predict(X_acc)

    if np.std(ig_resid) < 1e-10 or np.std(bench_resid) < 1e-10:
        r_partial, p_partial = float("nan"), float("nan")
    else:
        r_partial, p_partial = pearsonr(ig_resid, bench_resid)

    print(f"\n{'=' * 60}")
    print("CORRELATION RESULTS")
    print(f"{'=' * 60}")
    print(f"{'Model':<30} {'IG%':>6} {'BenchAcc':>9} {'BenchErr':>9}")
    for k in sorted(model_keys, key=lambda x: IG_SCORES[x]):
        print(f"  {k:<28} {IG_SCORES[k]:6.1f} {results[k]['accuracy']:9.3f} {1 - results[k]['accuracy']:9.3f}")

    print(f"\nPearson  r(IG, bench_error)  = {r_pearson:.3f}   p={p_pearson:.4f}")
    print(f"Spearman rho(IG, bench_error) = {r_spearman:.3f}   p={p_spearman:.4f}")
    print(f"Partial  r(IG, bench_error | FormInv_acc) = {r_partial:.3f}   p={p_partial:.4f}")

    # Conclusion
    if abs(r_pearson) >= 0.5 and p_pearson < 0.15:
        conclusion = "IG predicts benchmark performance beyond accuracy (moderate evidence)"
    elif abs(r_pearson) >= 0.3:
        conclusion = "IG shows weak-to-moderate correlation with benchmark error (inconclusive at n=9)"
    else:
        conclusion = "IG does not predict benchmark performance beyond accuracy"

    return {
        "pearson_r": round(float(r_pearson), 4),
        "pearson_p": round(float(p_pearson), 4),
        "spearman_rho": round(float(r_spearman), 4),
        "spearman_p": round(float(p_spearman), 4),
        "partial_r_controlling_accuracy": round(float(r_partial), 4),
        "partial_p": round(float(p_partial), 4),
        "n_models": len(model_keys),
        "model_order": model_keys,
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_all()
    correlations = compute_correlations(results)

    # Build output artifact
    per_model_summary = {}
    for ig_key, res in results.items():
        per_model_summary[ig_key] = {
            "accuracy": res["accuracy"],
            "n_correct": res["n_correct"],
            "n_total": res["n_total"],
            "n_invalid": res["n_invalid"],
        }

    artifact = {
        "benchmark": "mmlu_formal_logic",
        "benchmark_note": "FOLIO was gated; fell back to MMLU formal_logic (126 items -> 60 TRUE/FALSE items, 30 each, seed=42). TRUE items present correct answer; FALSE items present a randomly chosen incorrect answer.",
        "n_items": 60,
        "prompt_template": PROMPT_TEMPLATE,
        "per_model": per_model_summary,
        "correlation": correlations,
    }

    out_path = Path("artifacts/cross_benchmark_folio_results.json")
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nSaved results to {out_path}")

    # Print per-model table for markdown
    print("\nPer-model summary table:")
    print(f"{'Model':<30} {'IG%':>6} {'FormInv%':>9} {'Bench%':>9}")
    for ig_key in sorted(per_model_summary, key=lambda x: IG_SCORES.get(x, 99)):
        ig = IG_SCORES.get(ig_key, float("nan"))
        fi = FORMINV_ACC.get(ig_key, float("nan"))
        ba = per_model_summary[ig_key]["accuracy"] * 100
        print(f"  {ig_key:<28} {ig:6.1f} {fi:9.1f} {ba:9.1f}")
