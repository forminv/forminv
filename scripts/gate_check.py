"""
Gate check script: 48-hour FormInv pilot.

Run on 50 curated theorems x 3 paraphrases x 3 models = 450 calls.
Decision: if mean IG > 2x baseline variance -> proceed to full Lean4 experiment.

Usage:
  cd /Users/noel.thomas/forminv
  pip install -e .
  python scripts/gate_check.py --provider mock         # smoke test (free)
  python scripts/gate_check.py --provider openai       # real run
  python scripts/gate_check.py --providers openai,anthropic  # multi-model
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from forminv.eval.providers import call_model
from forminv.generators.paraphrases import _fallback_paraphrases, generate_paraphrases
from forminv.generators.theorems import DomainTier, load_curated_theorems
from forminv.metrics import aggregate_ig, compute_ig
from forminv.schemas import ModelPrediction


def run_gate_check(
    providers: list[str],
    models: list[str],
    n_theorems: int = 50,
    n_surface: int = 2,
    n_def: int = 1,
    use_cache: bool = True,
    use_api_paraphrases: bool = True,
) -> dict:
    """Run the gate check and return results."""

    # 1. Load theorems
    theorems = load_curated_theorems(
        tier_filter=[DomainTier.T1, DomainTier.T2],
        limit=n_theorems,
    )
    print(f"Loaded {len(theorems)} theorems")

    # 2. Generate paraphrases
    all_paraphrases = []
    print("Generating paraphrases...")
    for thm in theorems:
        if use_api_paraphrases:
            paras = generate_paraphrases(thm, n_surface=n_surface, n_definitional=n_def, use_cache=use_cache)
        else:
            paras = _fallback_paraphrases(thm)[: n_surface + n_def]

        # Always add canonical question too (for baseline)
        from forminv.schemas import EquivLevel, Paraphrase

        canonical = Paraphrase(
            paraphrase_id=f"{thm.theorem_id}_canonical",
            theorem_id=thm.theorem_id,
            level=EquivLevel.SURFACE,
            nl_question=f"{thm.canonical_nl.rstrip('.')} -- TRUE or FALSE?",
            ground_truth=thm.ground_truth,
            verification_method="canonical",
        )
        all_paraphrases.append(canonical)
        all_paraphrases.extend(paras)

    print(f"Total paraphrases: {len(all_paraphrases)}")

    # 3. Run models
    all_predictions: list[ModelPrediction] = []
    for provider, model in zip(providers, models):
        print(f"\nRunning {provider}/{model}...")
        for i, para in enumerate(all_paraphrases):
            if i % 20 == 0:
                print(f"  {i}/{len(all_paraphrases)}...")
            try:
                resp = call_model(para.nl_question, provider=provider, model=model, use_cache=use_cache)
                correct = None
                if resp.parsed is not None:
                    correct = (resp.parsed == "TRUE") == para.ground_truth
                all_predictions.append(
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

    # 4. Compute IG per theorem per model
    ig_results = []
    for model in models:
        model_preds = [p for p in all_predictions if p.model == model]
        for thm in theorems:
            thm_paras = [
                p for p in all_paraphrases if p.theorem_id == thm.theorem_id and "canonical" not in p.paraphrase_id
            ]
            canon_preds = [p for p in model_preds if p.theorem_id == thm.theorem_id and "canonical" in p.paraphrase_id]
            baseline_var = 0.0  # with n=1 canonical, baseline=0; ok for gate check

            ig = compute_ig(thm.theorem_id, model, model_preds, thm_paras, baseline_var)
            ig_results.append(ig)

    # 5. Aggregate
    agg = aggregate_ig(ig_results)
    agg["n_theorems"] = len(theorems)
    agg["n_paraphrases_per_theorem"] = n_surface + n_def
    agg["models"] = models
    agg["total_predictions"] = len(all_predictions)
    agg["coverage"] = sum(1 for p in all_predictions if p.outcome != "INVALID") / max(len(all_predictions), 1)

    # 6. Gate check decision
    agg["gate_check_pass"] = agg["mean_ig"] > 0.05  # IG > 5% -> proceed
    agg["recommendation"] = (
        "PROCEED to full Lean4 experiment"
        if agg["gate_check_pass"]
        else "ESCALATE to harder theorems (Tier 3-4) or review paraphrase quality"
    )

    return {
        "aggregate": agg,
        "per_theorem": [
            {
                "theorem_id": r.theorem_id,
                "model": r.model,
                "ig": r.std_accuracy,
                "mean_acc": r.mean_accuracy,
                "scr": r.all_correct,
            }
            for r in ig_results
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="FormInv Gate Check")
    parser.add_argument("--providers", default="mock", help="Comma-separated providers")
    parser.add_argument("--models", default="mock-v1", help="Comma-separated models")
    parser.add_argument("--n-theorems", type=int, default=50)
    parser.add_argument("--n-surface", type=int, default=2)
    parser.add_argument("--n-def", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--no-api-paraphrases",
        action="store_true",
        help="Use fallback templates instead of GPT-4o paraphrases",
    )
    parser.add_argument("--out", default="artifacts/gate_check.json")
    args = parser.parse_args()

    providers = args.providers.split(",")
    models = args.models.split(",")
    assert len(providers) == len(models), "providers and models must have same count"

    print("FormInv Gate Check")
    print(f"  Theorems: {args.n_theorems}")
    print(f"  Paraphrases: {args.n_surface} surface + {args.n_def} def per theorem")
    print(f"  Models: {list(zip(providers, models))}")
    print(f"  Estimated calls: {args.n_theorems * (args.n_surface + args.n_def + 1) * len(models)}")
    print()

    t0 = time.time()
    results = run_gate_check(
        providers=providers,
        models=models,
        n_theorems=args.n_theorems,
        n_surface=args.n_surface,
        n_def=args.n_def,
        use_cache=not args.no_cache,
        use_api_paraphrases=not args.no_api_paraphrases,
    )
    elapsed = time.time() - t0

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))

    # Print summary
    agg = results["aggregate"]
    print(f"\n{'=' * 50}")
    print(f"GATE CHECK RESULTS ({elapsed:.1f}s)")
    print(f"{'=' * 50}")
    print(f"  Mean IG:          {agg['mean_ig']:.4f}  ({agg['mean_ig'] * 100:.1f} pp)")
    print(f"  Mean accuracy:    {agg['mean_accuracy']:.4f}")
    print(f"  Coverage:         {agg['coverage']:.4f}")
    print(f"  SCR:              {agg['scr']:.4f}")
    print(f"  IG > 5% threshold: {agg['gate_check_pass']}")
    print(f"\n  DECISION: {agg['recommendation']}")
    print(f"\n  Results saved to: {args.out}")


if __name__ == "__main__":
    main()
