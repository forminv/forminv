"""
Cross-benchmark correlation: does FormInv IG predict performance on other benchmarks?

Hypothesis: models with higher FormInv IG on F5 (connective variation) will
perform worse on logical reasoning tasks (FOLIO, LogiQA, etc.)

We run a MINI version: 5 FOLIO-style starter items + measure correlation with FormInv IG.

NOTE: This is a starter set of 5 items. Expand to 20 before publication by adding
biconditional, transitivity, and contrapositive reasoning items from FOLIO/LogiQA.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 5 hand-crafted FOLIO-style logical reasoning items (starter set).
# These test biconditional/connective reasoning -- the same weakness as F5.
# TODO: expand to 20 items before publication.
FOLIO_MINI = [
    {
        "id": "folio_001",
        "question": (
            "All squares are rectangles. All rectangles have four sides. "
            "Does it follow that all squares have four sides?"
        ),
        "ground_truth": "TRUE",
        "family": "transitivity",
    },
    {
        "id": "folio_002",
        "question": (
            "A shape is a square if and only if it has four equal sides and four right angles. "
            "A shape has four equal sides and four right angles. "
            "Is it exactly the case that this shape is a square?"
        ),
        "ground_truth": "TRUE",
        "family": "biconditional",
    },
    {
        "id": "folio_003",
        "question": (
            "A number is prime precisely when it has exactly two positive divisors. "
            "Does 7 have exactly two positive divisors?"
        ),
        "ground_truth": "TRUE",
        "family": "biconditional",
    },
    {
        "id": "folio_004",
        "question": ("A number is even just in case it is divisible by 2. Is 6 divisible by 2?"),
        "ground_truth": "TRUE",
        "family": "biconditional",
    },
    {
        "id": "folio_005",
        "question": (
            "A function is continuous at x if and only if the limit of the function "
            "as it approaches x equals the function value at x. "
            "Is it the case that continuity at x requires the limit to equal the function value?"
        ),
        "ground_truth": "TRUE",
        "family": "biconditional_reverse",
    },
]


def run_cross_benchmark(providers: list[tuple], n_items: int | None = None) -> dict:
    """
    providers: list of (provider_str, model_str) tuples
    n_items: how many items from FOLIO_MINI to use (None = all)
    """
    from forminv.eval.providers import call_model

    items = FOLIO_MINI if n_items is None else FOLIO_MINI[:n_items]

    results = {}
    for provider, model in providers:
        print(f"\nEvaluating {provider}/{model} on {len(items)} FOLIO-mini items...")
        model_results = []
        for item in items:
            try:
                resp = call_model(item["question"], provider=provider, model=model)
                correct = (resp.parsed == item["ground_truth"]) if resp.parsed is not None else None
                model_results.append(
                    {
                        "item_id": item["id"],
                        "family": item["family"],
                        "response": resp.raw[:120],
                        "parsed": resp.parsed,
                        "correct": correct,
                    }
                )
                status = "OK" if correct else ("WRONG" if correct is False else "UNPARSED")
                print(f"  [{status}] {item['id']}: parsed={resp.parsed}")
            except Exception as e:
                print(f"  [ERROR] {provider}/{model} on {item['id']}: {e}")
                model_results.append(
                    {
                        "item_id": item["id"],
                        "family": item["family"],
                        "response": "",
                        "parsed": None,
                        "correct": None,
                        "error": str(e),
                    }
                )

        answered = [r for r in model_results if r["correct"] is not None]
        acc = sum(1 for r in answered if r["correct"]) / max(len(answered), 1)
        results[f"{provider}/{model}"] = {
            "accuracy": acc,
            "n_answered": len(answered),
            "n_total": len(model_results),
            "per_item": model_results,
        }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock provider (no API keys needed, always returns TRUE)",
    )
    parser.add_argument("--out", default="artifacts/cross_benchmark_results.json")
    args = parser.parse_args()

    if args.mock:
        providers = [
            ("mock", "mock-always-true"),
        ]
    else:
        providers = [
            ("anthropic", "claude-sonnet-4-6"),
            ("openai", "gpt-4o"),
            ("openai", "gpt-4o-mini"),
        ]

    results = run_cross_benchmark(providers)

    print("\n--- Cross-benchmark results ---")
    for model_key, res in results.items():
        print(f"  {model_key}: {res['accuracy'] * 100:.1f}% on {res['n_answered']}/{res['n_total']} FOLIO-mini items")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.out}")

    print("\nNote: cross-benchmark IG correlation requires FormInv IG scores (already computed).")
    print("Key comparison: IG on F5 (connective variation) vs accuracy on biconditional FOLIO items.")
    print("If models with higher IG also do worse on biconditional items, correlation is positive.")
