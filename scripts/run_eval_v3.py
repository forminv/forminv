"""
FormInv v3 full evaluation -- 8 families x N models.

Usage:
  python scripts/run_eval_v3.py \
    --dataset data/generated/forminv_v3_50.jsonl \
    --providers openai,anthropic,deepseek,gemini,openai \
    --models gpt-4o-mini,claude-sonnet-4-6,deepseek-chat,gemini-2.5-flash,gpt-4o \
    --out artifacts/v3_eval_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from forminv.eval.providers import call_model
from forminv.metrics import aggregate_ig, compute_cross_model_disagreement, compute_ig
from forminv.schemas import EquivLevel, ModelPrediction, Paraphrase


def load_dataset(path: str) -> tuple[list[dict], list[Paraphrase]]:
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))

    paraphrases = [
        Paraphrase(
            paraphrase_id=item["id"],
            theorem_id=item["theorem_id"],
            level=EquivLevel.SURFACE,
            nl_question=item["nl_question"],
            ground_truth=item["ground_truth"] == "TRUE",
            verification_method=item.get("verification_method", "unknown"),
        )
        for item in items
    ]
    return items, paraphrases


def run_all_models(
    dataset_path: str,
    providers: list[str],
    models: list[str],
    use_cache: bool = True,
    workers: int = 1,
) -> dict:
    items, paraphrases = load_dataset(dataset_path)
    theorem_ids = sorted(set(p.theorem_id for p in paraphrases))

    all_predictions: dict[str, list[ModelPrediction]] = {}

    for provider, model in zip(providers, models):
        model_key = f"{provider}/{model}"
        print(f"\n=== {model_key} ({len(paraphrases)} items) ===")
        preds = []
        for i, para in enumerate(paraphrases):
            if i % 50 == 0:
                print(f"  {i}/{len(paraphrases)}...")
            try:
                resp = call_model(para.nl_question, provider=provider, model=model, use_cache=use_cache)
                correct = None
                if resp.parsed is not None:
                    correct = (resp.parsed == "TRUE") == para.ground_truth
                preds.append(
                    ModelPrediction(
                        pred_id=f"{para.paraphrase_id}__{model}",
                        paraphrase_id=para.paraphrase_id,
                        theorem_id=para.theorem_id,
                        model=model,
                        provider=provider,
                        raw_response=resp.raw,
                        parsed_label=resp.parsed == "TRUE" if resp.parsed else None,
                        outcome=resp.parsed or "INVALID",
                        correct=correct,
                        latency_s=resp.latency_s,
                    )
                )
            except Exception as e:
                print(f"  [WARN] {para.paraphrase_id}: {e}")
        all_predictions[model_key] = preds

    # Compute per-family IG
    results = {}
    for model_key, preds in all_predictions.items():
        model = model_key.split("/")[1]
        ig_results = []
        for tid in theorem_ids:
            thm_paras = [p for p in paraphrases if p.theorem_id == tid and "canonical" not in p.paraphrase_id]
            ig = compute_ig(tid, model, preds, thm_paras)
            ig_results.append(ig)

        agg = aggregate_ig(ig_results)

        # Per-family failure rates
        family_failures: dict[str, list[float]] = {}
        for pred in preds:
            if pred.correct is not None and "canonical" not in pred.paraphrase_id:
                # Extract family from paraphrase_id
                fam = pred.paraphrase_id.split("_")[-1]
                family_failures.setdefault(fam, []).append(float(not pred.correct))

        agg["family_failure_rates"] = {k: float(np.mean(v)) for k, v in family_failures.items()}
        agg["coverage"] = sum(1 for p in preds if p.outcome != "INVALID") / max(len(preds), 1)
        results[model_key] = agg

    # Cross-model disagreements
    preds_by_model_short = {mk.split("/")[1]: v for mk, v in all_predictions.items()}
    disagreements = compute_cross_model_disagreement(preds_by_model_short, paraphrases)

    return {
        "per_model": results,
        "cross_model_disagreements": len(disagreements),
        "disagreement_examples": disagreements[:20],  # first 20
        "n_theorems": len(theorem_ids),
        "n_items": len(paraphrases),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--providers", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--out", default="artifacts/v3_eval_results.json")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    providers = args.providers.split(",")
    models = args.models.split(",")
    assert len(providers) == len(models)

    print("FormInv v3 Evaluation")
    print(f"  Dataset: {args.dataset}")
    print(f"  Models: {list(zip(providers, models))}")

    t0 = time.time()
    results = run_all_models(args.dataset, providers, models, use_cache=not args.no_cache)
    elapsed = time.time() - t0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 60}")
    print(f"V3 RESULTS ({elapsed:.0f}s)")
    print(f"{'=' * 60}")
    print(f"{'Model':<30} {'Mean IG':>8} {'Mean Acc':>10} {'SCR':>8} {'Coverage':>10}")
    print("-" * 70)
    for mk, agg in sorted(results["per_model"].items(), key=lambda x: x[1]["mean_ig"], reverse=True):
        print(
            f"{mk:<30} {agg['mean_ig'] * 100:>7.1f}%  {agg['mean_accuracy'] * 100:>8.1f}%  "
            f"{agg['scr'] * 100:>6.0f}%  {agg.get('coverage', 1) * 100:>8.0f}%"
        )

    print(f"\nCross-model disagreements: {results['cross_model_disagreements']}")
    print("\nPer-family failure rates (GPT-4o):")
    gpt4o_key = next((k for k in results["per_model"] if "gpt-4o" in k and "mini" not in k), None)
    if gpt4o_key:
        for fam, rate in sorted(
            results["per_model"][gpt4o_key].get("family_failure_rates", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            bar = "#" * int(rate * 20)
            print(f"  {fam:25} {rate * 100:5.1f}% {bar}")

    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
