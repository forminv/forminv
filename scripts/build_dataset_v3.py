"""
Build FormInv v3 dataset -- full scale.

Scale: 500 theorems x 8 families x 1 paraphrase + 1 canonical = 4,500 items
Models: 10 (configured in configs/v3_run.yaml)
Total API calls for dataset build: ~500 (paraphrase generation)
Total eval calls: ~45,000

Usage:
  cd /Users/noel.thomas/forminv
  # Step 1: Build paraphrase dataset (requires OpenAI key)
  python scripts/build_dataset_v3.py --n-theorems 200 --out data/generated/forminv_v3_200.jsonl

  # Step 2: Verify dataset
  python scripts/verify_dataset.py --dataset data/generated/forminv_v3_200.jsonl

  # Step 3: Run evaluation (see run_eval_v3.py)
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from forminv.generators.paraphrases_v2 import get_paraphrases_for_eval
from forminv.generators.theorems import DomainTier, load_curated_theorems


def build_dataset(n_theorems: int, out_path: str, use_cache: bool = True, tiers: list[str] | None = None) -> None:
    tier_filter = None
    if tiers:
        tier_map = {
            "t1": DomainTier.T1,
            "t2": DomainTier.T2,
            "t3": DomainTier.T3,
            "t4": DomainTier.T4,
        }
        tier_filter = [tier_map[t] for t in tiers if t in tier_map]

    print(f"Building FormInv v3 dataset: {n_theorems} theorems x 8 families + canonical")
    theorems = load_curated_theorems(tier_filter=tier_filter, limit=n_theorems)
    print(f"Loaded {len(theorems)} theorems")

    all_items = []
    errors = 0

    for i, thm in enumerate(theorems):
        if i % 10 == 0:
            print(f"  [{i}/{len(theorems)}] Generating paraphrases for {thm.theorem_id}...")
        try:
            paraphrases = get_paraphrases_for_eval(thm, include_canonical=True, use_cache=use_cache)
            for para in paraphrases:
                all_items.append(
                    {
                        "id": para.paraphrase_id,
                        "theorem_id": para.theorem_id,
                        "lean4_statement": thm.lean4_statement,
                        "mathlib_name": thm.mathlib_name,
                        "domain": thm.domain,
                        "tier": thm.tier.value,
                        "family": para.paraphrase_id.split("_")[-1] if "_" in para.paraphrase_id else "canonical",
                        "nl_question": para.nl_question,
                        "ground_truth": "TRUE" if para.ground_truth else "FALSE",
                        "canonical_nl": thm.canonical_nl,
                        "verification_method": para.verification_method,
                        "generation_seed": thm.seed,
                    }
                )
        except Exception as e:
            print(f"  [WARN] {thm.theorem_id}: {e}")
            errors += 1
            continue

    # Compute dataset hash
    content = json.dumps(all_items, sort_keys=True).encode()
    dataset_hash = hashlib.sha256(content).hexdigest()[:16]

    # Write dataset
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for item in all_items:
            f.write(json.dumps(item) + "\n")

    # Write manifest
    manifest = {
        "dataset_id": f"forminv_v3_{n_theorems}",
        "n_theorems": len(theorems),
        "n_items": len(all_items),
        "n_errors": errors,
        "families": sorted(set(item["family"] for item in all_items)),
        "domains": sorted(set(item["domain"] for item in all_items)),
        "tiers": sorted(set(item["tier"] for item in all_items)),
        "sha256": dataset_hash,
        "output": str(out),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nDataset built:")
    print(f"  Items: {len(all_items)}")
    print(f"  Errors: {errors}")
    print(f"  Families: {manifest['families']}")
    print(f"  SHA256: {dataset_hash}")
    print(f"  Output: {out}")
    print(f"  Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-theorems", type=int, default=50)
    parser.add_argument("--out", default="data/generated/forminv_v3.jsonl")
    parser.add_argument("--tiers", default="t1,t2", help="Comma-separated tiers to include")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    tiers = args.tiers.split(",") if args.tiers else None
    build_dataset(args.n_theorems, args.out, use_cache=not args.no_cache, tiers=tiers)


if __name__ == "__main__":
    main()
