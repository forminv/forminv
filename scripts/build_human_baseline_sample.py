"""
Sample 30 paraphrases for human baseline study.
Study design:
- 30 items from forminv_v3_103.jsonl
- Focus on F5 (connective variation) and F6 (comparison order) -- the hardest families
- Stratified: up to 15 variation items + up to 10 order items + up to 5 other families
- Question for annotators: "Are these two statements mathematically equivalent?"
- Expected: human kappa > 0.9 (these ARE equivalent by construction)
"""

import json
import random
from collections import Counter
from pathlib import Path


def main():
    random.seed(42)

    dataset = [json.loads(l) for l in open("data/generated/forminv_v3_103.jsonl")]

    # Separate by family using the explicit 'family' field (not ID substring matching)
    f5_items = [x for x in dataset if x["family"] == "variation"]
    f6_items = [x for x in dataset if x["family"] == "order"]
    other_items = [x for x in dataset if x["family"] not in ("canonical", "variation", "order")]

    print(f"Available: variation={len(f5_items)}, order={len(f6_items)}, other={len(other_items)}")

    # Build canonical lookup by theorem_id
    canonicals = {x["theorem_id"]: x for x in dataset if x["family"] == "canonical"}

    # Sample (capped by available items)
    sample = (
        random.sample(f5_items, min(15, len(f5_items)))
        + random.sample(f6_items, min(10, len(f6_items)))
        + random.sample(other_items, min(5, len(other_items)))
    )

    # Build study items
    study_items = []
    skipped = 0
    for item in sample:
        canonical = canonicals.get(item["theorem_id"])
        if not canonical:
            skipped += 1
            continue
        study_items.append(
            {
                "item_id": item["id"],
                "theorem_id": item["theorem_id"],
                "family": item["family"],
                "statement_A": canonical["nl_question"],
                "statement_B": item["nl_question"],
                "ground_truth_equivalent": True,  # ALL paraphrases are equivalent by construction
                "annotator_answer": None,  # to be filled by human
                "annotator_confidence": None,  # 1-3 scale: 1=guessing, 2=fairly sure, 3=certain
            }
        )

    if skipped > 0:
        print(f"Warning: {skipped} items skipped (canonical not found for their theorem_id)")

    out = Path("data/human_baseline/study_sample_30.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(
        json.dumps(
            {
                "metadata": {
                    "n_items": len(study_items),
                    "families": dict(Counter(x["family"] for x in study_items)),
                    "ground_truth_all_equivalent": True,
                    "random_seed": 42,
                },
                "instructions": (
                    "For each pair of mathematical statements, answer: "
                    "Are A and B mathematically equivalent? "
                    "They express the same mathematical claim if they are both TRUE (or both FALSE) "
                    "for all mathematical objects satisfying the stated conditions. "
                    "Answer: YES/NO. "
                    "Confidence: 1=guessing, 2=fairly sure, 3=certain."
                ),
                "items": study_items,
            },
            indent=2,
        )
    )

    print(f"Created {len(study_items)} study items at {out}")
    print("\nFamily breakdown:")
    counts = Counter(x["family"] for x in study_items)
    for fam, n in sorted(counts.items()):
        print(f"  {fam}: {n}")


if __name__ == "__main__":
    main()
