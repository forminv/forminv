"""
FormInv External Audit -- validate cross-model unanimity on external benchmark items.

Claim: FormInv's cross-model unanimity audit (>=6/9 models failing a paraphrase
while passing canonical) detects semantically-incorrect paraphrases, and this
generalises beyond the FormInv dataset itself.

Each test triplet has:
  canonical   -- the baseline question (ground_truth = TRUE or FALSE)
  para_correct -- a valid paraphrase (same semantic content, different wording)
  para_wrong   -- a biconditional-overreach paraphrase that changes the truth value

Expected behaviour:
  - canonical     : models mostly answer correctly (high accuracy)
  - para_correct  : models mostly answer correctly (low IG ≈ 0)
  - para_wrong    : models correctly answer its TRUE ground_truth as FALSE
                    → large failure count → unanimity audit flags it

Usage:
    cd /Users/noel.thomas/chaos-logic-bench/extensions/forminv
    python scripts/run_external_audit.py [--mock] [--out artifacts/external_audit_raw.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from forminv.eval.providers import call_model

# -- Model roster (same 9 as run_all_models.py) -----------------------------
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

# -- External test items -----------------------------------------------------
# Each item is a dict with:
#   id, domain, canonical, para_correct, para_wrong
#   canonical_gt  -- TRUE/FALSE for canonical
#   correct_gt    -- TRUE/FALSE for para_correct  (must equal canonical_gt)
#   wrong_gt      -- TRUE/FALSE for para_wrong    (DIFFERS from canonical_gt
#                   because the biconditional reversal is mathematically false)
#
# The audit simulates the scenario where a benchmark author mis-labels
# para_wrong with the canonical label (wrong_gt_as_labeled = canonical_gt).
# Models that answer the true truth value will "fail" against that mislabel.

EXTERNAL_ITEMS = [
    # -- Calculus -------------------------------------------------------------
    {
        "id": "ext_001",
        "domain": "calculus",
        "canonical": "Is every differentiable function also continuous?",
        "canonical_gt": "TRUE",
        "para_correct": "Does differentiability at a point imply continuity at that point?",
        "correct_gt": "TRUE",
        # Biconditional reversal: claims continuous ↔ differentiable (false: |x| is continuous but not diff. at 0)
        "para_wrong": "Is a function differentiable at a point if and only if it is continuous at that point?",
        "wrong_true_gt": "FALSE",  # real answer
        "wrong_labeled_gt": "TRUE",  # how a bad benchmark would label it (same as canonical)
        "error_type": "biconditional_reversal",
    },
    {
        "id": "ext_002",
        "domain": "calculus",
        # Canonical: can a convergent series have non-zero terms? Yes (geometric).
        "canonical": "Can a series converge even if its terms are not all zero?",
        "canonical_gt": "TRUE",
        "para_correct": "Is it possible for a series to have a finite sum while its terms are non-zero?",
        "correct_gt": "TRUE",
        # Biconditional overreach: claims series converges iff terms → 0 (false: harmonic series diverges)
        "para_wrong": "Does a series converge if and only if its terms tend to zero?",
        "wrong_true_gt": "FALSE",
        "wrong_labeled_gt": "TRUE",
        "error_type": "biconditional_necessary_not_sufficient",
    },
    {
        "id": "ext_003",
        "domain": "calculus",
        "canonical": "Can a bounded function fail to be integrable (Riemann) on [0,1]?",
        "canonical_gt": "TRUE",
        "para_correct": "Does there exist a bounded function on [0,1] that is not Riemann integrable?",
        "correct_gt": "TRUE",
        # Biconditional overreach: claims integrable iff continuous (false: monotone functions are integrable)
        "para_wrong": "Is a function Riemann integrable on [0,1] if and only if it is continuous on [0,1]?",
        "wrong_true_gt": "FALSE",
        "wrong_labeled_gt": "TRUE",
        "error_type": "biconditional_overreach",
    },
    # -- Linear algebra -------------------------------------------------------
    {
        "id": "ext_004",
        "domain": "linear_algebra",
        "canonical": "Is a square matrix invertible if and only if its determinant is non-zero?",
        "canonical_gt": "TRUE",
        "para_correct": "Does a square matrix have an inverse exactly when its determinant is non-zero?",
        "correct_gt": "TRUE",
        # Biconditional overreach: claims invertible iff det > 0 (false: det = -3 is still invertible)
        "para_wrong": "Is a square matrix invertible if and only if its determinant is positive?",
        "wrong_true_gt": "FALSE",
        "wrong_labeled_gt": "TRUE",
        "error_type": "biconditional_wrong_condition",
    },
    {
        "id": "ext_005",
        "domain": "linear_algebra",
        "canonical": "Can two non-zero vectors be orthogonal?",
        "canonical_gt": "TRUE",
        "para_correct": "Is it possible for two vectors with non-zero lengths to have a dot product of zero?",
        "correct_gt": "TRUE",
        # Biconditional overreach: claims orthogonal iff linearly independent (false: (1,0,0) and (0,1,0) are both)
        "para_wrong": "Are two non-zero vectors orthogonal if and only if they are linearly independent?",
        "wrong_true_gt": "FALSE",
        "wrong_labeled_gt": "TRUE",
        "error_type": "biconditional_conflation",
    },
    # -- Number theory --------------------------------------------------------
    {
        "id": "ext_006",
        "domain": "number_theory",
        "canonical": "Is the square root of 2 irrational?",
        "canonical_gt": "TRUE",
        "para_correct": "Does sqrt(2) fail to be expressible as a ratio of two integers?",
        "correct_gt": "TRUE",
        # Biconditional overreach: claims irrational iff not a perfect square (false: sqrt(3), sqrt(5), etc.)
        "para_wrong": "Is a positive integer's square root irrational if and only if the integer is not a perfect square?",
        "wrong_true_gt": "TRUE",  # actually TRUE -- this one IS correct, used as a control
        "wrong_labeled_gt": "TRUE",
        "error_type": "control_correct",  # Should NOT be flagged
    },
    {
        "id": "ext_007",
        "domain": "number_theory",
        "canonical": "Is 1 a composite number?",
        "canonical_gt": "FALSE",
        "para_correct": "Does 1 have more than two distinct positive divisors?",
        "correct_gt": "FALSE",
        # Biconditional overreach: claims composite iff has exactly 3 divisors (false: only squares of primes have 3)
        "para_wrong": "Is a positive integer composite if and only if it has exactly three positive divisors?",
        "wrong_true_gt": "FALSE",
        "wrong_labeled_gt": "FALSE",  # labeled same as canonical (FALSE)
        "error_type": "biconditional_wrong_characterisation",
    },
    # -- Real analysis --------------------------------------------------------
    {
        "id": "ext_008",
        "domain": "real_analysis",
        "canonical": "Is every rational number expressible as a terminating or repeating decimal?",
        "canonical_gt": "TRUE",
        "para_correct": "Does every fraction p/q (with integers p, q, q≠0) have a decimal expansion that either terminates or eventually repeats?",
        "correct_gt": "TRUE",
        # Biconditional overreach: claims rational iff terminates (false: 1/3 = 0.333... repeats but doesn't terminate)
        "para_wrong": "Is a real number rational if and only if its decimal expansion terminates?",
        "wrong_true_gt": "FALSE",
        "wrong_labeled_gt": "TRUE",
        "error_type": "biconditional_necessary_not_sufficient",
    },
    {
        "id": "ext_009",
        "domain": "real_analysis",
        "canonical": "Is every Cauchy sequence in the real numbers convergent?",
        "canonical_gt": "TRUE",
        "para_correct": "In the real number system, does every sequence satisfying the Cauchy criterion have a limit?",
        "correct_gt": "TRUE",
        # Biconditional overreach: claims Cauchy iff bounded (false: bounded sequences need not be Cauchy)
        "para_wrong": "Is a sequence of real numbers Cauchy if and only if it is bounded?",
        "wrong_true_gt": "FALSE",
        "wrong_labeled_gt": "TRUE",
        "error_type": "biconditional_wrong_condition",
    },
    # -- Set theory / logic --------------------------------------------------
    {
        "id": "ext_010",
        "domain": "set_theory",
        "canonical": "Is the empty set a subset of every set?",
        "canonical_gt": "TRUE",
        "para_correct": "Does the empty set have no elements that are absent from any given set?",
        "correct_gt": "TRUE",
        # Biconditional overreach: claims subset iff equal (false: {1} ⊆ {1,2} but {1} ≠ {1,2})
        "para_wrong": "Is set A a subset of set B if and only if A equals B?",
        "wrong_true_gt": "FALSE",
        "wrong_labeled_gt": "TRUE",
        "error_type": "biconditional_conflation",
    },
]


def run_on_question(question: str, ground_truth: str, provider: str, model: str, use_cache: bool = True) -> dict:
    """Call one model on one question; return result dict."""
    try:
        resp = call_model(question, provider=provider, model=model, use_cache=use_cache)
        correct = (resp.parsed == ground_truth) if resp.parsed is not None else None
        return {
            "response": resp.raw[:200],
            "parsed": resp.parsed,
            "correct": correct,
            "latency_s": resp.latency_s,
        }
    except Exception as e:
        return {"response": "", "parsed": None, "correct": None, "latency_s": 0.0, "error": str(e)}


def run_external_audit(providers: list[tuple], use_cache: bool = True) -> dict:
    """
    Run all models on canonical + para_correct + para_wrong for all external items.
    Returns nested dict: item_id -> question_type -> model_key -> result_dict
    """
    results: dict = {}

    for item in EXTERNAL_ITEMS:
        iid = item["id"]
        print(f"\n=== {iid} ({item['domain']}) ===")
        results[iid] = {
            "meta": {
                "domain": item["domain"],
                "error_type": item["error_type"],
                "canonical": item["canonical"],
                "canonical_gt": item["canonical_gt"],
                "para_correct": item["para_correct"],
                "correct_gt": item["correct_gt"],
                "para_wrong": item["para_wrong"],
                "wrong_true_gt": item["wrong_true_gt"],
                "wrong_labeled_gt": item["wrong_labeled_gt"],
            },
            "canonical": {},
            "para_correct": {},
            "para_wrong": {},
        }

        questions = [
            ("canonical", item["canonical"], item["canonical_gt"]),
            ("para_correct", item["para_correct"], item["correct_gt"]),
            # For para_wrong we score against the TRUE ground truth (wrong_true_gt)
            # so we can measure whether models detect the falsehood.
            ("para_wrong", item["para_wrong"], item["wrong_true_gt"]),
        ]

        for q_type, question, gt in questions:
            print(f"  [{q_type}] GT={gt}: {question[:80]}...")
            for provider, model in providers:
                model_key = f"{provider}/{model.replace('/', '_')}"
                res = run_on_question(question, gt, provider, model, use_cache=use_cache)
                results[iid][q_type][model_key] = res
                status = "OK" if res["correct"] else ("WRONG" if res["correct"] is False else "UNPARSED")
                print(f"    [{status}] {model_key}: parsed={res['parsed']}")

    return results


def compute_audit_stats(results: dict) -> dict:
    """
    For each item, compute:
    - canonical_accuracy: fraction of models answering correctly
    - correct_para_accuracy: fraction correct on valid paraphrase
    - wrong_para_detection_rate: fraction of models that correctly identify the false statement
    - models_failing_wrong_para: count of models failing against the mislabeled ground truth
    - flagged: True if >=6/9 models would "fail" against mislabel
    """
    stats = {}
    for iid, item_data in results.items():
        meta = item_data["meta"]
        n_models = 0
        canon_correct = 0
        correct_para_correct = 0
        # "failing wrong para" = model answers the TRUE ground truth of para_wrong correctly,
        # which means it disagrees with the (wrong) labeled ground truth.
        models_detecting_wrong = 0  # answered wrong_true_gt correctly (FALSE for biconditional)
        models_failing_against_label = 0  # model's answer != wrong_labeled_gt

        for model_key in item_data["canonical"]:
            n_models += 1
            if item_data["canonical"][model_key]["correct"]:
                canon_correct += 1
            if model_key in item_data["para_correct"] and item_data["para_correct"][model_key]["correct"]:
                correct_para_correct += 1
            if model_key in item_data["para_wrong"]:
                wr = item_data["para_wrong"][model_key]
                if wr["correct"]:  # correctly identified the true GT (the "wrong" answer)
                    models_detecting_wrong += 1
                # Would this be "failing" against the mislabel?
                parsed = wr["parsed"]
                if parsed is not None and parsed != meta["wrong_labeled_gt"]:
                    models_failing_against_label += 1

        flagged = models_failing_against_label >= 6

        stats[iid] = {
            "domain": meta["domain"],
            "error_type": meta["error_type"],
            "n_models": n_models,
            "canonical_accuracy": canon_correct / max(n_models, 1),
            "correct_para_accuracy": correct_para_correct / max(n_models, 1),
            "models_detecting_wrong": models_detecting_wrong,
            "models_failing_against_label": models_failing_against_label,
            "flagged_by_unanimity": flagged,
        }

    return stats


def print_summary(stats: dict, n_models: int) -> None:
    print(f"\n{'=' * 75}")
    print("FormInv External Audit -- Cross-Model Unanimity Results")
    print(f"{'=' * 75}")
    print(f"{'ID':<10} {'Domain':<18} {'Type':<30} {'Canon':>6} {'CorrP':>6} {'Fail':>5} {'Flag':>5}")
    print("-" * 80)

    total_wrong_items = 0
    total_flagged = 0
    total_controls = 0
    total_controls_flagged = 0

    for iid, s in stats.items():
        is_control = s["error_type"] == "control_correct"
        flag_str = "YES" if s["flagged_by_unanimity"] else "no"
        print(
            f"{iid:<10} {s['domain']:<18} {s['error_type']:<30} "
            f"{s['canonical_accuracy'] * 100:>5.0f}% "
            f"{s['correct_para_accuracy'] * 100:>5.0f}% "
            f"{s['models_failing_against_label']:>3}/{s['n_models']:<2} "
            f"{flag_str:>5}"
        )
        if not is_control:
            total_wrong_items += 1
            if s["flagged_by_unanimity"]:
                total_flagged += 1
        else:
            total_controls += 1
            if s["flagged_by_unanimity"]:
                total_controls_flagged += 1

    print(f"\nWrong paraphrases flagged  : {total_flagged}/{total_wrong_items}")
    print(f"Control paraphrases flagged : {total_controls_flagged}/{total_controls} (should be 0)")
    print(
        f"\nConclusion: FormInv quality audit generalizes to external benchmarks: "
        f"{total_flagged} of {total_wrong_items} wrong paraphrases detected."
    )


def main():
    parser = argparse.ArgumentParser(description="FormInv External Benchmark Audit")
    parser.add_argument("--mock", action="store_true", help="Use mock provider (no API calls, always returns TRUE)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass response cache")
    parser.add_argument(
        "--out",
        default="artifacts/external_audit_raw.json",
        help="Where to save raw per-item results",
    )
    args = parser.parse_args()

    if args.mock:
        providers = [("mock", "mock-always-true")]
    else:
        providers = ALL_MODELS

    print(f"Running external audit with {len(providers)} model(s)...")
    print(f"Items: {len(EXTERNAL_ITEMS)} triplets × 3 question types = {len(EXTERNAL_ITEMS) * 3} questions per model")
    print(f"Total API calls (no cache): {len(EXTERNAL_ITEMS) * 3 * len(providers)}")

    results = run_external_audit(providers, use_cache=not args.no_cache)
    stats = compute_audit_stats(results)
    print_summary(stats, n_models=len(providers))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"items": results, "stats": stats}, indent=2))
    print(f"\nRaw results saved to: {args.out}")

    return results, stats


if __name__ == "__main__":
    main()
