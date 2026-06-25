"""
FormInv -- Run ALL models with correct provider keys.
Green-lit by researcher: use all available keys.

Provider routing:
- Anthropic models: ANTHROPIC_API_KEY
- OpenAI models: OPENAI_API_KEY
- DeepSeek models: DEEPSEEK_API_KEY
- Gemini models: GOOGLE_API_KEY (or OpenRouter fallback)
- Others: OPENROUTER_API_KEY

Usage:
  cd /Users/noel.thomas/forminv
  python scripts/run_all_models.py --dataset data/generated/forminv_v3_50.jsonl
  python scripts/run_all_models.py --dataset data/generated/forminv_v3_103.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from forminv.eval.providers import call_model
from forminv.metrics import aggregate_ig, compute_cross_model_disagreement, compute_ig
from forminv.schemas import EquivLevel, ModelPrediction, Paraphrase

# -- Model registry -- all models we want to run ------------------------------
ALL_MODELS = [
    # OpenAI
    ("openai", "gpt-4o", "flagship"),
    ("openai", "gpt-4o-mini", "small"),
    ("openai", "o4-mini", "reasoning"),  # OpenAI reasoning model
    # Anthropic
    ("anthropic", "claude-sonnet-4-6", "flagship"),
    ("anthropic", "claude-haiku-4-5", "small"),
    # DeepSeek (direct key)
    ("deepseek", "deepseek-chat", "flagship"),  # DeepSeek V3
    ("deepseek", "deepseek-reasoner", "reasoning"),  # DeepSeek R1
    # Gemini (Google key)
    ("gemini", "gemini-2.5-flash", "flagship"),
    # OpenRouter (for models without direct keys)
    ("openrouter", "meta-llama/llama-3.3-70b-instruct", "open_source"),
]


def fix_provider(provider: str, model: str) -> tuple[str, str]:
    """Route to correct provider based on model name."""
    if "deepseek" in model.lower() and provider == "openrouter":
        return "deepseek", model
    if "gemini" in model.lower() and os.environ.get("GOOGLE_API_KEY"):
        return "gemini", model
    return provider, model


def load_dataset(path: str) -> list[Paraphrase]:
    items = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            items.append(
                Paraphrase(
                    paraphrase_id=item["id"],
                    theorem_id=item["theorem_id"],
                    level=EquivLevel.SURFACE,
                    nl_question=item["nl_question"],
                    ground_truth=item["ground_truth"] == "TRUE",
                    verification_method=item.get("verification_method", ""),
                )
            )
    return items


def run_model(
    paraphrases: list[Paraphrase], provider: str, model: str, use_cache: bool = True
) -> list[ModelPrediction]:
    """Run one model on all paraphrases."""
    preds = []
    for i, para in enumerate(paraphrases):
        if i % 50 == 0:
            print(f"  [{i}/{len(paraphrases)}] {provider}/{model}...")
        try:
            resp = call_model(para.nl_question, provider=provider, model=model, use_cache=use_cache)
            correct = None
            if resp.parsed is not None:
                correct = (resp.parsed == "TRUE") == para.ground_truth
            preds.append(
                ModelPrediction(
                    pred_id=f"{para.paraphrase_id}__{model.replace('/', '_')}",
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
    return preds


def analyze_results(all_predictions: dict[str, list[ModelPrediction]], paraphrases: list[Paraphrase]) -> dict:
    """Compute IG, per-family breakdown, and cross-model disagreements."""
    theorem_ids = sorted(set(p.theorem_id for p in paraphrases))
    results = {}

    for model_key, preds in all_predictions.items():
        if not preds:
            continue
        model = model_key.split("/")[1]
        coverage = sum(1 for p in preds if p.outcome != "INVALID") / max(len(preds), 1)
        if coverage < 0.5:
            print(f"  [SKIP] {model_key}: coverage={coverage:.1%} (too many invalid responses)")
            continue

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
                fam = pred.paraphrase_id.split("_")[-1]
                family_failures.setdefault(fam, []).append(float(not pred.correct))

        agg["family_failure_rates"] = {k: float(np.mean(v)) for k, v in family_failures.items()}
        agg["coverage"] = coverage
        results[model_key] = agg

    # Cross-model disagreement
    preds_short = {mk.split("/")[1]: v for mk, v in all_predictions.items()}
    disagreements = compute_cross_model_disagreement(preds_short, paraphrases)

    return {
        "per_model": results,
        "cross_model_disagreements": len(disagreements),
        "disagreement_examples": disagreements[:20],
    }


def print_table(results: dict) -> None:
    print(f"\n{'=' * 70}")
    print("FORMINV RESULTS -- ALL MODELS")
    print(f"{'=' * 70}")
    print(f"{'Model':<35} {'IG':>7} {'Acc':>7} {'SCR':>7} {'Cov':>7}")
    print("-" * 60)
    for mk, agg in sorted(results["per_model"].items(), key=lambda x: x[1]["mean_ig"], reverse=True):
        print(
            f"{mk:<35} {agg['mean_ig'] * 100:>6.1f}% "
            f"{agg['mean_accuracy'] * 100:>6.1f}% "
            f"{agg['scr'] * 100:>6.0f}% "
            f"{agg.get('coverage', 1) * 100:>6.0f}%"
        )

    print(f"\nCross-model disagreements: {results['cross_model_disagreements']}")

    # Worst family overall
    print(f"\n{'Family':<20} {'Avg Failure':>12}")
    print("-" * 35)
    fam_agg: dict[str, list[float]] = {}
    for mk, agg in results["per_model"].items():
        for fam, rate in agg.get("family_failure_rates", {}).items():
            fam_agg.setdefault(fam, []).append(rate)
    for fam, rates in sorted(fam_agg.items(), key=lambda x: np.mean(x[1]), reverse=True):
        bar = "#" * int(np.mean(rates) * 20)
        print(f"{fam:<20} {np.mean(rates) * 100:>10.1f}% {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/generated/forminv_v3_50.jsonl")
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    parser.add_argument("--out", default="artifacts/all_models_results.json")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    paraphrases = load_dataset(args.dataset)
    print(f"Loaded {len(paraphrases)} paraphrases from {args.dataset}")

    models_to_run = ALL_MODELS
    if args.models != "all":
        selected = set(args.models.split(","))
        models_to_run = [(p, m, t) for p, m, t in ALL_MODELS if m in selected]

    all_predictions = {}
    for provider, model, tier in models_to_run:
        prov, mod = fix_provider(provider, model)
        model_key = f"{prov}/{mod}"
        print(f"\n=== {model_key} ({tier}) ===")
        preds = run_model(paraphrases, prov, mod, use_cache=not args.no_cache)
        all_predictions[model_key] = preds

    results = analyze_results(all_predictions, paraphrases)
    print_table(results)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
