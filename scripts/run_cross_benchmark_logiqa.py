"""
Cross-benchmark correlation experiment -- LogiQA arm.

Runs all 9 FormInv models on:
  - MMLU formal_logic  (logical reasoning, 60 items)
  - MMLU global_facts  (factual recall -- negative control, 60 items)

Computes per-model accuracy, then correlates against FormInv IG scores.

Usage:
    cd /Users/noel.thomas/forminv
    .venv/bin/python scripts/run_cross_benchmark_logiqa.py [--mock]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from forminv.eval.providers import call_model_mc

# --- Model registry ------------------------------------------------------------
MODELS = [
    ("deepseek", "deepseek-chat"),
    ("openai", "gpt-4o"),
    ("openai", "o4-mini"),
    ("gemini", "gemini-2.5-flash"),
    ("openai", "gpt-4o-mini"),
    ("deepseek", "deepseek-reasoner"),
    ("anthropic", "claude-sonnet-4-6"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    ("anthropic", "claude-haiku-4-5"),
]

# --- FormInv IG and accuracy (from artifacts/all_8models_results.json) ---------
FORMINV_IG = {
    "deepseek-chat": 6.93,
    "gpt-4o": 6.97,
    "o4-mini": 7.59,
    "gemini-2.5-flash": 7.69,
    "gpt-4o-mini": 8.67,
    "deepseek-reasoner": 8.84,
    "claude-sonnet-4-6": 10.72,
    "meta-llama/llama-3.3-70b-instruct": 13.75,
    "claude-haiku-4-5": 19.38,
}

FORMINV_ACC = {
    "deepseek-chat": 96.43,
    "gpt-4o": 94.24,
    "o4-mini": 96.37,
    "gemini-2.5-flash": 94.90,
    "gpt-4o-mini": 94.29,
    "deepseek-reasoner": 93.02,
    "claude-sonnet-4-6": 93.17,
    "meta-llama/llama-3.3-70b-instruct": 89.80,
    "claude-haiku-4-5": 86.04,
}

ANSWER_LETTERS = ["A", "B", "C", "D"]


def run_benchmark(items: list[dict], provider: str, model: str, use_mock: bool = False) -> dict:
    """Run one model on a list of MMLU items. Returns accuracy stats."""
    results = []
    n_correct = 0
    n_answered = 0

    for i, item in enumerate(items):
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]  # 0-indexed int
        correct_letter = ANSWER_LETTERS[answer_idx]

        try:
            if use_mock:
                # Mock: always pick A
                from forminv.eval.providers import LLMResponse

                resp = LLMResponse(raw="A", parsed="A", latency_s=0.001)
            else:
                resp = call_model_mc(question, choices, provider, model)

            answered = resp.parsed is not None
            correct = resp.parsed == correct_letter if answered else False

            if answered:
                n_answered += 1
            if correct:
                n_correct += 1

            results.append(
                {
                    "idx": i,
                    "question": question[:80] + "..." if len(question) > 80 else question,
                    "correct_letter": correct_letter,
                    "predicted": resp.parsed,
                    "raw": resp.raw[:40],
                    "correct": correct,
                    "latency_s": resp.latency_s,
                }
            )

            if (i + 1) % 10 == 0:
                running_acc = n_correct / (i + 1) * 100
                print(f"    [{i + 1}/{len(items)}] running acc={running_acc:.1f}%")

        except Exception as e:
            print(f"    ERROR on item {i}: {e}")
            results.append(
                {
                    "idx": i,
                    "question": question[:80],
                    "correct_letter": correct_letter,
                    "predicted": None,
                    "raw": f"ERROR: {e}",
                    "correct": False,
                    "latency_s": 0.0,
                }
            )

    accuracy = n_correct / len(items) if items else 0.0
    error_rate = 1.0 - accuracy

    return {
        "accuracy": accuracy,
        "error_rate": error_rate,
        "n_correct": n_correct,
        "n_answered": n_answered,
        "n_total": len(items),
        "per_item": results,
    }


def compute_correlations(model_stats: dict, benchmark_key: str) -> dict:
    """Compute Pearson, Spearman, and partial correlations."""
    import numpy as np
    from scipy import stats

    model_names = []
    ig_scores = []
    error_rates = []
    forminv_accs = []

    for model_name, bmarks in model_stats.items():
        if benchmark_key not in bmarks:
            continue
        ig = FORMINV_IG.get(model_name)
        fa = FORMINV_ACC.get(model_name)
        if ig is None or fa is None:
            continue
        err = bmarks[benchmark_key]["error_rate"] * 100  # as percentage

        model_names.append(model_name)
        ig_scores.append(ig)
        error_rates.append(err)
        forminv_accs.append(fa)

    n = len(model_names)
    if n < 3:
        return {"error": f"Only {n} models, need >=3"}

    ig_arr = np.array(ig_scores)
    err_arr = np.array(error_rates)
    fa_arr = np.array(forminv_accs)

    # Pearson
    pearson_r, pearson_p = stats.pearsonr(ig_arr, err_arr)

    # Spearman
    spearman_r, spearman_p = stats.spearmanr(ig_arr, err_arr)

    # Jackknife: remove Haiku (highest IG outlier) and recompute
    haiku_idx = None
    for i, nm in enumerate(model_names):
        if "haiku" in nm.lower():
            haiku_idx = i
            break
    jk_result = {}
    if haiku_idx is not None and n >= 4:
        mask = [i for i in range(n) if i != haiku_idx]
        ig_jk = ig_arr[mask]
        err_jk = err_arr[mask]
        if len(ig_jk) >= 3 and np.std(ig_jk) > 0 and np.std(err_jk) > 0:
            pr_jk, pp_jk = stats.pearsonr(ig_jk, err_jk)
            sr_jk, sp_jk = stats.spearmanr(ig_jk, err_jk)
            jk_result = {
                "pearson_r": float(pr_jk),
                "pearson_p": float(pp_jk),
                "spearman_r": float(sr_jk),
                "spearman_p": float(sp_jk),
                "n": len(mask),
                "note": "Haiku excluded",
            }

    # Partial correlation: r(IG, error | FormInv_acc)
    # Residualize both IG and error on FormInv_acc, then correlate residuals
    partial_result = {}
    try:
        # Regress IG on FormInv_acc; get residuals
        slope_ig, intercept_ig, _, _, _ = stats.linregress(fa_arr, ig_arr)
        resid_ig = ig_arr - (slope_ig * fa_arr + intercept_ig)

        # Regress error on FormInv_acc; get residuals
        slope_err, intercept_err, _, _, _ = stats.linregress(fa_arr, err_arr)
        resid_err = err_arr - (slope_err * fa_arr + intercept_err)

        if np.std(resid_ig) > 1e-10 and np.std(resid_err) > 1e-10:
            part_r, part_p = stats.pearsonr(resid_ig, resid_err)
            partial_result = {
                "partial_r": float(part_r),
                "partial_p": float(part_p),
                "note": "r(IG, error | FormInv_accuracy)",
            }
    except Exception as e:
        partial_result = {"error": str(e)}

    return {
        "n": n,
        "models": model_names,
        "ig_scores": ig_scores,
        "error_rates": error_rates,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "jackknife_without_haiku": jk_result,
        "partial_correlation": partial_result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use mock provider for testing")
    parser.add_argument("--models", default="all", help="Comma-separated model names to run, or 'all'")
    parser.add_argument("--benchmark", default="both", help="formal_logic, global_facts, or both")
    args = parser.parse_args()

    base = Path(__file__).parent.parent

    # Load datasets
    benchmarks = {}
    if args.benchmark in ("formal_logic", "both"):
        with open(base / "data/raw/mmlu_formal_logic_60.json") as f:
            benchmarks["formal_logic"] = json.load(f)
        print(f"Loaded {len(benchmarks['formal_logic'])} formal_logic items")

    if args.benchmark in ("global_facts", "both"):
        with open(base / "data/raw/mmlu_global_facts_60.json") as f:
            benchmarks["global_facts"] = json.load(f)
        print(f"Loaded {len(benchmarks['global_facts'])} global_facts items")

    # Filter models if requested
    models_to_run = MODELS
    if args.models != "all":
        requested = set(args.models.split(","))
        models_to_run = [(p, m) for p, m in MODELS if m in requested]

    # Results accumulator: { model_name: { benchmark: {accuracy, error_rate, ...} } }
    out_path = base / "artifacts/cross_benchmark_logiqa_results.json"

    # Load existing results if any (for resumability)
    if out_path.exists() and not args.mock:
        with open(out_path) as f:
            existing = json.load(f)
        model_stats = existing.get("model_stats", {})
        print(f"Resuming from existing results ({len(model_stats)} models done)")
    else:
        model_stats = {}

    total_models = len(models_to_run)
    for mi, (provider, model) in enumerate(models_to_run):
        model_short = model.split("/")[-1] if "/" in model else model
        key = model  # use full model name as key (matches FORMINV_IG dict)
        print(f"\n[{mi + 1}/{total_models}] {provider}/{model}")

        if key not in model_stats:
            model_stats[key] = {}

        for bname, items in benchmarks.items():
            if bname in model_stats.get(key, {}):
                print(f"  {bname}: already done (acc={model_stats[key][bname]['accuracy'] * 100:.1f}%), skipping")
                continue

            print(f"  Running {bname} ({len(items)} items)...")
            t0 = time.time()
            result = run_benchmark(items, provider, model, use_mock=args.mock)
            elapsed = time.time() - t0
            print(
                f"  {bname}: acc={result['accuracy'] * 100:.1f}% ({result['n_correct']}/{result['n_total']}) in {elapsed:.1f}s"
            )

            model_stats[key][bname] = result

            # Save after each model/benchmark pair
            with open(out_path, "w") as f:
                json.dump({"model_stats": model_stats}, f, indent=2)
            print(f"  Saved to {out_path}")

    # Compute correlations
    print("\n=== Computing correlations ===")
    correlations = {}
    for bname in benchmarks:
        corr = compute_correlations(model_stats, bname)
        correlations[bname] = corr
        print(f"\n{bname}:")
        print(f"  Pearson r(IG, error) = {corr.get('pearson_r', '?'):.3f} (p={corr.get('pearson_p', '?'):.3f})")
        print(f"  Spearman rho         = {corr.get('spearman_r', '?'):.3f} (p={corr.get('spearman_p', '?'):.3f})")
        jk = corr.get("jackknife_without_haiku", {})
        if jk:
            print(f"  Jackknife (no Haiku) r={jk.get('pearson_r', '?'):.3f}, rho={jk.get('spearman_r', '?'):.3f}")
        pc = corr.get("partial_correlation", {})
        if pc and "partial_r" in pc:
            print(f"  Partial r(IG,error|FormInv_acc) = {pc['partial_r']:.3f} (p={pc['partial_p']:.3f})")

    # Save final results
    final = {
        "model_stats": model_stats,
        "correlations": correlations,
        "forminv_ig": FORMINV_IG,
        "forminv_acc": FORMINV_ACC,
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nFinal results saved to {out_path}")

    return model_stats, correlations


if __name__ == "__main__":
    main()
