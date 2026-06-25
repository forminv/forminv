"""
Compute Fleiss's kappa for the 9-model x 366-item agreement matrix.

All 9 models x 366 items are 100% cached -- no live API calls.

Formula:
  P_i  = (n_T*(n_T-1) + n_F*(n_F-1)) / (n*(n-1))
  P    = mean of P_i over all items
  p_j  = (total ratings in category j) / (N * n)
  P_e  = sum of p_j2
  kappa    = (P - P_e) / (1 - P_e)

Interpretation bands (Landis & Koch 1977):
  < 0.00  : poor
  0.00-0.20: slight
  0.21-0.40: fair
  0.41-0.60: moderate
  0.61-0.80: substantial
  0.81-1.00: almost perfect

Usage:
  cd /Users/noel.thomas/chaos-logic-bench/extensions/forminv
  python scripts/compute_fleiss_kappa.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from forminv.eval.providers import call_model

DATASET = Path(__file__).parent.parent / "data/generated/forminv_v3_50.jsonl"
OUT = Path(__file__).parent.parent / "artifacts/fleiss_kappa.json"

MODELS_9 = [
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


def interpret(kappa: float) -> str:
    if kappa < 0.00:
        return "poor"
    if kappa < 0.21:
        return "slight"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.61:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost perfect"


def fleiss_kappa(rating_matrix: list[list[int | None]]) -> dict:
    """
    rating_matrix: list of N items; each item is a list of n rater labels
                   (1=TRUE, 0=FALSE, None=invalid/skip).
    Items where any rater returned None are dropped (reported separately).
    Returns dict with kappa, P_bar, Pe_bar, n_items_used, n_items_dropped.
    """
    n_raters = len(rating_matrix[0])

    # Drop items with any None response
    complete = [row for row in rating_matrix if None not in row]
    dropped = len(rating_matrix) - len(complete)
    N = len(complete)
    n = n_raters

    if N == 0:
        raise ValueError("No complete items after dropping None responses.")

    # P_i per item
    P_list = []
    for row in complete:
        n_T = sum(row)
        n_F = n - n_T
        P_i = (n_T * (n_T - 1) + n_F * (n_F - 1)) / (n * (n - 1))
        P_list.append(P_i)

    P_bar = sum(P_list) / N

    # Overall proportions
    total_ratings = N * n
    total_true = sum(sum(row) for row in complete)
    p_T = total_true / total_ratings
    p_F = 1.0 - p_T
    P_e = p_T**2 + p_F**2

    if abs(1.0 - P_e) < 1e-12:
        kappa = float("nan")
    else:
        kappa = (P_bar - P_e) / (1.0 - P_e)

    return {
        "kappa": round(kappa, 4),
        "P_bar": round(P_bar, 4),
        "P_e": round(P_e, 4),
        "p_TRUE": round(p_T, 4),
        "p_FALSE": round(p_F, 4),
        "n_raters": n,
        "n_items_used": N,
        "n_items_dropped": dropped,
        "interpretation": interpret(kappa),
    }


def run():
    items = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    print(f"Dataset: {len(items)} items x {len(MODELS_9)} models")

    # Collect per-item labels: rating_matrix[item_idx][model_idx] = 1/0/None
    rating_matrix: list[list[int | None]] = []

    print("\nFetching labels from cache...")
    for i, item in enumerate(items):
        if i % 50 == 0:
            print(f"  {i}/{len(items)}...")
        row: list[int | None] = []
        for provider, model in MODELS_9:
            try:
                resp = call_model(item["nl_question"], provider=provider, model=model, use_cache=True)
                if resp.parsed == "TRUE":
                    row.append(1)
                elif resp.parsed == "FALSE":
                    row.append(0)
                else:
                    row.append(None)
            except Exception as e:
                print(f"  [WARN] item {i}, {model}: {e}")
                row.append(None)
        rating_matrix.append(row)

    print(f"  Done. {len(rating_matrix)} items x {len(MODELS_9)} models collected.")

    # Sanity check: unanimous-agreement synthetic test
    _test_all_agree = [[1] * 9] * 10 + [[0] * 9] * 10
    _k_perfect = fleiss_kappa(_test_all_agree)
    assert abs(_k_perfect["kappa"] - 1.0) < 1e-6, f"Sanity fail: {_k_perfect}"

    # Sanity check: random-ish labels -> kappa near 0
    import random

    random.seed(42)
    _test_random = [[random.randint(0, 1) for _ in range(9)] for _ in range(200)]
    _k_rand = fleiss_kappa(_test_random)
    assert abs(_k_rand["kappa"]) < 0.15, f"Sanity fail random: {_k_rand}"

    result = fleiss_kappa(rating_matrix)

    print(f"\n{'=' * 55}")
    print(f"Fleiss's kappa on 9-model x {result['n_items_used']}-item matrix")
    print(f"{'=' * 55}")
    print(f"  kappa             = {result['kappa']:.4f}  ({result['interpretation']})")
    print(f"  P (observed)  = {result['P_bar']:.4f}")
    print(f"  P_e (chance)  = {result['P_e']:.4f}")
    print(f"  p_TRUE        = {result['p_TRUE']:.4f}")
    print(f"  p_FALSE       = {result['p_FALSE']:.4f}")
    print(f"  Items used    = {result['n_items_used']}")
    print(f"  Items dropped = {result['n_items_dropped']}  (any rater returned None)")
    print(f"{'=' * 55}")

    # Model-level stats for context
    print("\nPer-model TRUE rates (context):")
    model_keys = [f"{p}/{m}" for p, m in MODELS_9]
    for j, mk in enumerate(model_keys):
        labels = [row[j] for row in rating_matrix if row[j] is not None]
        t_rate = sum(labels) / len(labels) if labels else float("nan")
        print(f"  {mk:<48} TRUE={t_rate:.1%}  n={len(labels)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "dataset": str(DATASET),
                "models": [f"{p}/{m}" for p, m in MODELS_9],
                **result,
            },
            indent=2,
        )
    )
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    run()
