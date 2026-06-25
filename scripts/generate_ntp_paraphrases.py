"""Generate 8-family paraphrases for ntp-mathlib theorems converted to FormInv format.

Reads a JSONL file of converted theorems (from convert_ntp_to_formInv.py),
constructs BaseTheorem objects, and calls the same paraphrase generation
pipeline used for curated theorems.

Usage:
  .venv/bin/python scripts/generate_ntp_paraphrases.py \\
    --theorems data/generated/ntp_mathlib_theorem_v1.jsonl \\
    --out data/generated/forminv_ntp30_paraphrases.jsonl \\
    --n 30
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from forminv.generators.paraphrases_v2 import get_paraphrases_for_eval
from forminv.schemas import BaseTheorem, DomainTier

TIER_MAP = {
    "T1": DomainTier.T1,
    "T2": DomainTier.T2,
    "T3": DomainTier.T3,
    "T4": DomainTier.T4,
    "tier1": DomainTier.T1,
    "tier2": DomainTier.T2,
    "tier3": DomainTier.T3,
    "tier4": DomainTier.T4,
}


def load_theorems(path: str, n: int) -> list[BaseTheorem]:
    theorems = []
    for line in open(path):
        if not line.strip():
            continue
        t = json.loads(line)
        tier = TIER_MAP.get(t.get("tier", "T1"), DomainTier.T1)
        bt = BaseTheorem(
            theorem_id=t["theorem_id"],
            lean4_statement=t["lean4_statement"],
            canonical_nl=t["canonical_nl"],
            ground_truth=t.get("ground_truth", True),
            domain=t.get("domain", "other"),
            tier=tier,
            mathlib_name=t["mathlib_name"],
            cas_verifiable=t.get("cas_verifiable", False),
            seed=t.get("seed", 42),
        )
        theorems.append(bt)
        if len(theorems) >= n:
            break
    return theorems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theorems", default="data/generated/ntp_mathlib_theorem_v1.jsonl")
    parser.add_argument("--out", default="data/generated/forminv_ntp30_paraphrases.jsonl")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    theorems = load_theorems(args.theorems, args.n)
    print(f"Loaded {len(theorems)} theorems from {args.theorems}")

    all_items = []
    errors = 0

    for i, thm in enumerate(theorems):
        print(f"  [{i + 1}/{len(theorems)}] {thm.theorem_id[:60]}")
        try:
            paraphrases = get_paraphrases_for_eval(
                thm,
                include_canonical=True,
                use_cache=not args.no_cache,
            )
            for para in paraphrases:
                family = para.paraphrase_id.split("_")[-1] if "_" in para.paraphrase_id else "canonical"
                all_items.append(
                    {
                        "id": para.paraphrase_id,
                        "theorem_id": para.theorem_id,
                        "lean4_statement": thm.lean4_statement,
                        "mathlib_name": thm.mathlib_name,
                        "domain": thm.domain,
                        "tier": thm.tier.value,
                        "family": family,
                        "nl_question": para.nl_question,
                        "ground_truth": "TRUE" if para.ground_truth else "FALSE",
                        "canonical_nl": thm.canonical_nl,
                        "verification_method": para.verification_method,
                        "generation_seed": thm.seed,
                        "source": "ntp-mathlib",
                    }
                )
        except Exception as e:
            print(f"     [WARN] {thm.theorem_id}: {e}")
            errors += 1
            continue

    # Compute hash
    content = json.dumps(all_items, sort_keys=True).encode()
    dataset_hash = hashlib.sha256(content).hexdigest()[:16]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for item in all_items:
            f.write(json.dumps(item) + "\n")

    # Write manifest
    manifest = {
        "dataset_id": f"forminv_ntp_{len(theorems)}",
        "n_theorems": len(theorems),
        "n_items": len(all_items),
        "n_errors": errors,
        "families": sorted(set(item["family"] for item in all_items)),
        "domains": sorted(set(item["domain"] for item in all_items)),
        "tiers": sorted(set(item["tier"] for item in all_items)),
        "sha256": dataset_hash,
        "output": str(out),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "ntp-mathlib",
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nDataset built:")
    print(f"  Items: {len(all_items)}")
    print(f"  Errors: {errors}")
    print(f"  Families: {manifest['families']}")
    print(f"  Domains: {manifest['domains']}")
    print(f"  SHA256: {dataset_hash}")
    print(f"  Output: {out}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
