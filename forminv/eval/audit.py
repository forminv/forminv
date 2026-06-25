"""FormInv audit -- paraphrase quality audit against cross-model disagreement.

Flags paraphrase items that too few models fail, which are candidates for being
semantically transparent (too easy) and worth manual review.
"""

import json
from pathlib import Path


def run_audit(args) -> None:
    """Flag paraphrase items where fewer than ``threshold`` models fail them.

    Counts, per item, how many models answered it incorrectly across the
    ``per_item`` results, then prints the items failed by fewer than
    ``args.threshold`` models. Items that very few models fail are candidates
    for being too easy (semantically transparent paraphrases) and should be
    reviewed. Falls back to a cross-model disagreement summary when no per-item
    data is present.

    Args:
        args: Parsed CLI arguments with ``results`` (path to the eval results
            JSON) and ``threshold`` (int minimum failing-model count) attributes.

    Returns:
        None.

    Raises:
        FileNotFoundError: If ``args.results`` does not exist.
    """
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {args.results}")

    with open(results_path) as f:
        results = json.load(f)

    per_model = results.get("per_model", {})
    n_models = len(per_model)
    threshold = args.threshold

    print("FormInv Paraphrase Quality Audit")
    print(f"  Results  : {args.results}")
    print(f"  Models   : {n_models}")
    print(f"  Threshold: {threshold} (flag items where <{threshold} models fail)")
    print()

    # Collect per-item failure counts across models
    item_failures: dict[str, int] = {}
    item_keys: set[str] = set()

    for model_key in per_model:
        item_results = results.get("per_item", {}).get(model_key, {})
        for item_id, item_data in item_results.items():
            item_keys.add(item_id)
            if isinstance(item_data, dict):
                failed = not item_data.get("correct", True)
            else:
                failed = not bool(item_data)
            if failed:
                item_failures[item_id] = item_failures.get(item_id, 0) + 1

    if not item_keys:
        print("No per-item data found in results -- cannot run item-level audit.")
        print("(Per-item audit requires results produced by forminv eval with per_item tracking.)")
        # Fall back to cross-model disagreement summary if available
        cmd = results.get("cross_model_disagreements")
        if cmd is not None:
            print(f"\nCross-model disagreements: {cmd}")
        return

    flagged = []
    for item_id in sorted(item_keys):
        n_failed = item_failures.get(item_id, 0)
        if n_failed < threshold:
            flagged.append((item_id, n_failed))

    print(f"Items audited : {len(item_keys)}")
    print(f"Items flagged : {len(flagged)} (failed by <{threshold}/{n_models} models)")
    print()

    if flagged:
        print(f"{'Item ID':<40} {'Models failing':>15}")
        print("-" * 57)
        for item_id, n_failed in flagged:
            print(f"{item_id:<40} {n_failed:>10}/{n_models}")
    else:
        print("No items flagged -- all items challenged by enough models.")
