"""
FALSE controls pilot -- run Haiku and Sonnet on 25 FALSE items.
Reports:
  - per-model accuracy (should be high for good models)
  - per-theorem consistency across 5 paraphrases
  - always-TRUE oracle baseline (= 50/75 = 66.7% balanced with 50 TRUE + 25 FALSE)
Usage:
  cd /Users/noel.thomas/chaos-logic-bench/extensions/forminv
  python scripts/run_false_controls_pilot.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from forminv.eval.providers import call_model

DATASET = Path(__file__).parent.parent / "data/generated/false_controls_pilot.jsonl"
OUT = Path(__file__).parent.parent / "artifacts/false_controls_pilot_results.json"

MODELS = [
    ("anthropic", "claude-haiku-4-5"),
    ("anthropic", "claude-sonnet-4-6"),
]


def run():
    items = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(items)} FALSE items")
    print(f"Theorem IDs: {sorted(set(i['theorem_id'] for i in items))}\n")

    results = {}
    for provider, model in MODELS:
        model_key = f"{provider}/{model}"
        print(f"\n=== {model_key} ===")
        preds = []
        for item in items:
            q = item["nl_question"]
            gt = item["ground_truth"]  # "FALSE" for all items
            try:
                resp = call_model(q, provider=provider, model=model, use_cache=True)
                parsed = resp.parsed  # "TRUE" or "FALSE" or None
                correct = (parsed == gt) if parsed else None
                preds.append(
                    {
                        "id": item["id"],
                        "theorem_id": item["theorem_id"],
                        "family": item["family"],
                        "gt": gt,
                        "parsed": parsed,
                        "correct": correct,
                        "raw": resp.raw[:80],
                    }
                )
                status = "+" if correct else ("-" if correct is False else "?")
                print(f"  {status} {item['id'][:50]}: pred={parsed}")
            except Exception as e:
                print(f"  [ERR] {item['id']}: {e}")
                preds.append(
                    {
                        "id": item["id"],
                        "theorem_id": item["theorem_id"],
                        "family": item["family"],
                        "gt": gt,
                        "parsed": None,
                        "correct": None,
                        "error": str(e),
                    }
                )

        # Compute stats
        valid = [p for p in preds if p["correct"] is not None]
        n_correct = sum(1 for p in valid if p["correct"])
        accuracy = n_correct / len(valid) if valid else 0

        # Consistency per theorem (answers same across paraphrases)
        by_theorem = defaultdict(list)
        for p in preds:
            by_theorem[p["theorem_id"]].append(p["parsed"])
        consistent = sum(1 for v in by_theorem.values() if len(set(v)) == 1 and None not in set(v))
        scr_false = consistent / len(by_theorem)

        # TRUE-bias: fraction answering TRUE (should be low on FALSE items)
        true_rate = sum(1 for p in valid if p["parsed"] == "TRUE") / len(valid) if valid else 0

        print(f"\n  --- {model} on 25 FALSE items ---")
        print(f"  Accuracy (correct=FALSE): {n_correct}/{len(valid)} = {accuracy:.1%}")
        print(f"  TRUE-bias: {true_rate:.1%} answered TRUE (should be ~0% on FALSE items)")
        print(f"  Consistency (SCR on FALSE): {consistent}/{len(by_theorem)} theorems = {scr_false:.1%}")

        results[model_key] = {
            "preds": preds,
            "n_items": len(items),
            "n_valid": len(valid),
            "accuracy_on_false": accuracy,
            "true_bias_rate": true_rate,
            "scr_false": scr_false,
            "per_theorem": {
                tid: {
                    "answers": answers,
                    "consistent": len(set(answers)) == 1 and None not in set(answers),
                }
                for tid, answers in by_theorem.items()
            },
        }

    # Always-TRUE oracle stats:
    # On 50 TRUE items: always-TRUE gets 100% = 50/50
    # On 25 FALSE items: always-TRUE gets 0% = 0/25
    # Balanced accuracy = (100% + 0%) / 2 = 50.0%
    # But in main dataset (50T+25F=75 items):
    # oracle_accuracy = 50/75 = 66.7%
    oracle_balanced_acc = 0.50  # per-class average
    oracle_raw_acc = 50 / 75  # fraction correct if only TRUE items exist

    print("\n\n=== ALWAYS-TRUE ORACLE BASELINE ===")
    print(f"  Raw accuracy on 75 items (50T+25F): {oracle_raw_acc:.1%}")
    print(f"  Balanced accuracy (average of TRUE-class and FALSE-class): {oracle_balanced_acc:.1%}")

    # Print per-model comparison
    print("\n=== SUMMARY ===")
    print(f"{'Model':<40} {'Acc(FALSE)':<15} {'TRUE-bias':<12} {'SCR(FALSE)'}")
    for mkey, r in results.items():
        print(f"{mkey:<40} {r['accuracy_on_false']:<15.1%} {r['true_bias_rate']:<12.1%} {r['scr_false']:.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "oracle_balanced_accuracy": oracle_balanced_acc,
                "oracle_raw_accuracy_75items": oracle_raw_acc,
                "models": results,
            },
            indent=2,
        )
    )
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    run()
