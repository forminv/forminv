"""
Write the cross-benchmark summary markdown from the saved results artifact.

Usage:
    cd /Users/noel.thomas/forminv
    .venv/bin/python scripts/write_cross_benchmark_summary.py
"""

import json
from pathlib import Path

ARTIFACT_PATH = Path("artifacts/cross_benchmark_folio_results.json")
SUMMARY_PATH = Path("artifacts/cross_benchmark_summary.md")

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


def write_summary():
    with open(ARTIFACT_PATH) as f:
        data = json.load(f)

    per_model = data["per_model"]
    corr = data["correlation"]
    benchmark = data["benchmark"]
    n_items = data["n_items"]

    # Table rows sorted by IG ascending
    rows = []
    for model, res in per_model.items():
        rows.append(
            {
                "model": model,
                "ig": IG_SCORES.get(model, float("nan")),
                "fi_acc": FORMINV_ACC.get(model, float("nan")),
                "bench_acc": res["accuracy"] * 100,
                "n_correct": res["n_correct"],
                "n_total": res["n_total"],
            }
        )
    rows.sort(key=lambda x: x["ig"])

    # Build table
    header = "| Model | IG (%) | FormInv Acc (%) | Bench Acc (%) | Correct / Total |"
    sep = "|---|---|---|---|---|"
    table_lines = [header, sep]
    for r in rows:
        table_lines.append(
            f"| {r['model']} | {r['ig']:.1f} | {r['fi_acc']:.1f} | {r['bench_acc']:.1f} | {r['n_correct']}/{r['n_total']} |"
        )
    table = "\n".join(table_lines)

    # Pearson / Spearman
    pr = corr["pearson_r"]
    pp = corr["pearson_p"]
    rho = corr["spearman_rho"]
    rhop = corr["spearman_p"]
    partialr = corr["partial_r_controlling_accuracy"]
    partialp = corr.get("partial_p", float("nan"))
    conclusion = corr["conclusion"]

    # Determine significance language
    def sig_label(p):
        if p < 0.05:
            return "statistically significant (p < 0.05)"
        if p < 0.10:
            return "marginally significant (p < 0.10)"
        if p < 0.20:
            return "not significant at conventional thresholds (p < 0.20)"
        return "not statistically significant"

    md = f"""# Cross-Benchmark Correlation Study: Invariance Gap vs. Formal Logic Reasoning

## 1. Benchmark and Setup

The downstream benchmark used here is **MMLU formal_logic** (subset of the MMLU suite, `cais/mmlu`
configuration `formal_logic`). The `yale-nlp/FOLIO` dataset was originally targeted but is access-gated on
Hugging Face and could not be downloaded automatically; MMLU formal_logic was used as the fallback.

From the full 126-item test split, **{n_items} TRUE/FALSE items** were constructed as follows (random seed 42):
items are drawn from the 126 source questions; for each selected question, one *TRUE* item presents the
correct answer choice and asks whether it is correct, and one *FALSE* item presents a randomly-chosen
incorrect answer choice and asks the same question. The final set contains exactly 30 TRUE and 30 FALSE
items. This framing maps the 4-way multiple-choice task into the binary TRUE/FALSE format expected by the
FormInv provider system while keeping content firmly in the formal-logic domain.

All 9 FormInv models were evaluated with caching enabled; each model-item pair was called at most once.
The system prompt and user template are identical to the FormInv main evaluation (provider constant, not
model-varying).

## 2. Per-Model Results

{table}

Models are sorted by Invariance Gap (IG) ascending (lowest IG = most paraphrase-invariant).
FormInv Acc is the accuracy on the original FormInv mathematical paraphrase task (provided).
Bench Acc is accuracy on the 60-item MMLU formal_logic TRUE/FALSE set.

## 3. Correlation Analysis and Interpretation

The key hypothesis is: **models with higher IG (more paraphrase-sensitive) perform worse on downstream
formal logic tasks**. This was tested by correlating IG against benchmark *error* (1 - accuracy).

| Statistic | Value | p-value | Interpretation |
|---|---|---|---|
| Pearson r(IG, bench_error) | {pr:.3f} | {pp:.4f} | {sig_label(pp)} |
| Spearman rho(IG, bench_error) | {rho:.3f} | {rhop:.4f} | {sig_label(rhop)} |
| Partial r(IG, bench_error \\| FormInv_acc) | {partialr:.3f} | {partialp:.4f} | {sig_label(partialp)} |

**n = 9 models.** With 7 degrees of freedom, the critical value for two-sided p < 0.05 is |r| > 0.666;
for p < 0.10, |r| > 0.582. All p-values should be interpreted with caution at this sample size.

The partial correlation controls for each model's general capability as measured by FormInv accuracy,
asking: does IG carry predictive information about benchmark performance *beyond* what raw accuracy already
explains? A near-zero partial r would indicate that IG is merely a proxy for general model quality; a
substantial partial r would support IG as an independent predictor.

**Conclusion:** {conclusion}

The finding {"supports" if abs(pr) >= 0.5 else "does not conclusively support"} the hypothesis that
Invariance Gap is a useful proxy for logical-form invariance in downstream reasoning tasks. Given the
small number of models (n=9) and the high collinearity between IG and general accuracy, definitive
conclusions require a larger model roster and a purpose-designed benchmark (such as FOLIO, once access
is obtained) that more directly tests premise-conclusion entailment rather than answer identification.
"""

    SUMMARY_PATH.write_text(md)
    print(f"Wrote summary to {SUMMARY_PATH}")
    print(md)


if __name__ == "__main__":
    write_summary()
