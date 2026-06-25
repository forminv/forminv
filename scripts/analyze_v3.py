"""
FormInv v3 analysis script -- generates paper-ready tables and figures.

Usage:
  python scripts/analyze_v3.py --results artifacts/v3_eval_5models.json
"""

import argparse
import json

import numpy as np


def analyze(results_path: str) -> None:
    with open(results_path) as f:
        results = json.load(f)

    per_model = results["per_model"]

    print("=" * 70)
    print("FORMINV V3 ANALYSIS")
    print("=" * 70)

    # -- Table 1: Main results by model ---------------------------------
    print("\n### TABLE 1: Model Comparison (sorted by Mean IG)")
    print(f"{'Model':<30} {'Mean IG':>8} {'Max IG':>8} {'Acc':>8} {'SCR':>6} {'Hi-IG%':>8}")
    print("-" * 72)
    for mk, agg in sorted(per_model.items(), key=lambda x: x[1]["mean_ig"], reverse=True):
        print(
            f"{mk:<30} {agg['mean_ig'] * 100:>7.1f}%  {agg['max_ig'] * 100:>7.1f}%  "
            f"{agg['mean_accuracy'] * 100:>6.1f}%  {agg['scr'] * 100:>5.0f}%  "
            f"{agg.get('pct_high_ig_10', 0) * 100:>7.0f}%"
        )

    print("\n  Hi-IG% = fraction of theorems with IG > 10%")
    print("  SCR = Semantic Consistency Rate (all paraphrases correct)")

    # -- Table 2: Per-family failure rates ------------------------------
    print("\n### TABLE 2: Per-Family Failure Rates (% wrong answers by family)")
    all_families = set()
    for agg in per_model.values():
        all_families.update(agg.get("family_failure_rates", {}).keys())

    families = sorted(all_families)
    models = list(per_model.keys())

    header = f"{'Family':<28}" + "".join(f"{m.split('/')[1][:10]:>12}" for m in models)
    print(header)
    print("-" * (28 + 12 * len(models)))

    for fam in families:
        row = f"{fam:<28}"
        for mk in models:
            rate = per_model[mk].get("family_failure_rates", {}).get(fam, float("nan"))
            if np.isnan(rate):
                row += f"{'N/A':>12}"
            else:
                row += f"{rate * 100:>11.1f}%"
        print(row)

    # -- Table 3: Cross-model disagreement ------------------------------
    print("\n### TABLE 3: Cross-Model Disagreements")
    print(f"  Total disagreements: {results.get('cross_model_disagreements', 0)}")

    disagrees = results.get("disagreement_examples", [])
    if disagrees:
        print("  Sample disagreements:")
        for d in disagrees[:5]:
            print(f"    {d['paraphrase_id'][:50]}")
            print(f"      {d['model_a']}: {d['label_a']} | {d['model_b']}: {d['label_b']} (GT: {d['ground_truth']})")

    # -- Key finding: Accuracy vs IG correlation -------------------------
    accs = [v["mean_accuracy"] for v in per_model.values()]
    igs = [v["mean_ig"] for v in per_model.values()]
    if len(accs) > 2:
        corr = np.corrcoef(accs, igs)[0, 1]
        print("\n### KEY FINDING: Accuracy-IG Correlation")
        print(f"  r(mean_accuracy, mean_IG) = {corr:.3f}")
        if corr > 0.3:
            print("  POSITIVE: higher accuracy -> higher IG (more capable = more overconfident)")
        elif corr < -0.3:
            print("  NEGATIVE: higher accuracy -> lower IG (capable models are more consistent)")
        else:
            print("  NEAR ZERO: accuracy and IG are decoupled")

    # -- Worst families (highest failure) -------------------------------
    print("\n### WORST FAMILIES (highest average failure rate across models)")
    family_avg = {}
    for fam in families:
        rates = [per_model[mk].get("family_failure_rates", {}).get(fam) for mk in models]
        rates = [r for r in rates if r is not None]
        if rates:
            family_avg[fam] = np.mean(rates)

    for fam, rate in sorted(family_avg.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(rate * 30)
        print(f"  {fam:<25} {rate * 100:5.1f}% {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    analyze(args.results)


if __name__ == "__main__":
    main()
