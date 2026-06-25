"""
FALSE controls -- all 8 models (Llama/OpenRouter skipped per user instruction).

Models:
  anthropic/claude-haiku-4-5      (cached)
  anthropic/claude-sonnet-4-6     (cached)
  openai/gpt-4o                   (live API)
  openai/gpt-4o-mini              (live API)
  openai/o4-mini                  (live API)
  deepseek/deepseek-chat          (live API)
  deepseek/deepseek-reasoner      (live API)
  gemini/gemini-2.5-flash         (live API)

Usage:
  cd /Users/noel.thomas/chaos-logic-bench/extensions/forminv
  python scripts/run_false_controls_all_models.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from forminv.eval.providers import call_model

DATASET = Path(__file__).parent.parent / "data/generated/false_controls_pilot.jsonl"
OUT = Path(__file__).parent.parent / "artifacts/false_controls_all_models.json"

MODELS = [
    ("anthropic", "claude-haiku-4-5"),
    ("anthropic", "claude-sonnet-4-6"),
    ("openai", "gpt-4o"),
    ("openai", "gpt-4o-mini"),
    ("openai", "o4-mini"),
    ("deepseek", "deepseek-chat"),
    ("deepseek", "deepseek-reasoner"),
    ("gemini", "gemini-2.5-flash"),
]

# always-TRUE oracle on this 25-FALSE-item set:
#   accuracy = 0/25 = 0.0%  (every item is FALSE, so TRUE is always wrong)
#   balanced-acc over {TRUE-class, FALSE-class} in the full 75-item set:
#     TRUE-class acc = 50/50 = 100%, FALSE-class acc = 0/25 = 0%  -> balanced = 50%
ORACLE_ACC_ON_FALSE = 0.0
ORACLE_BALANCED_ACC = 0.5


def model_stats(preds: list[dict]) -> dict:
    valid = [p for p in preds if p["parsed"] is not None]
    n = len(valid)
    n_correct = sum(1 for p in valid if p["correct"])
    accuracy = n_correct / n if n else 0.0
    true_rate = sum(1 for p in valid if p["parsed"] == "TRUE") / n if n else 0.0

    by_theorem: dict[str, list] = defaultdict(list)
    for p in preds:
        by_theorem[p["theorem_id"]].append(p["parsed"])
    consistent = sum(1 for v in by_theorem.values() if len(set(v)) == 1 and None not in set(v))
    scr_false = consistent / len(by_theorem) if by_theorem else 0.0
    return {
        "n_items": len(preds),
        "n_valid": n,
        "accuracy_on_false": accuracy,
        "true_bias_rate": true_rate,
        "scr_false": scr_false,
        "beats_always_true": accuracy > ORACLE_ACC_ON_FALSE,
    }


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
            gt = item["ground_truth"]  # always "FALSE"
            try:
                resp = call_model(q, provider=provider, model=model, use_cache=True)
                parsed = resp.parsed
                correct = (parsed == gt) if parsed is not None else None
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
                print(f"  {status} {item['id'][:55]}: pred={parsed}")
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

        stats = model_stats(preds)
        print(
            f"  Acc(FALSE)={stats['accuracy_on_false']:.1%}  "
            f"TRUE-bias={stats['true_bias_rate']:.1%}  "
            f"SCR(FALSE)={stats['scr_false']:.1%}  "
            f"beats-always-TRUE={stats['beats_always_true']}"
        )
        results[model_key] = {"preds": preds, **stats}

    # ---- summary table ----
    print("\n\n" + "=" * 75)
    print("FALSE Controls -- All 8 Models")
    print("=" * 75)
    print(f"{'Model':<42} {'Acc(FALSE)':>10} {'TRUE-bias':>10} {'SCR(FALSE)':>11} {'Beats always-TRUE?':>18}")
    print("-" * 75)
    for mk, r in results.items():
        beats = "YES" if r["beats_always_true"] else "NO"
        print(
            f"{mk:<42} {r['accuracy_on_false']:>9.1%} {r['true_bias_rate']:>10.1%} {r['scr_false']:>10.1%} {beats:>18}"
        )
    print("-" * 75)
    print(f"{'always-TRUE oracle (balanced-acc=50%)':<42} {'0.0%':>10} {'100.0%':>10} {'n/a':>11} {'baseline':>18}")
    print("=" * 75)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "oracle_balanced_accuracy": ORACLE_BALANCED_ACC,
                "oracle_acc_on_false_items": ORACLE_ACC_ON_FALSE,
                "note": "Llama (openrouter) skipped -- OpenRouter not approved for this run.",
                "models": results,
            },
            indent=2,
        )
    )
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    run()
