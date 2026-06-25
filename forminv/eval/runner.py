"""FormInv eval runner -- delegates to ``scripts/run_eval_v3.py`` logic.

Thin CLI-facing wrapper that resolves the model roster (the built-in
:data:`ALL_MODELS` set or a user-supplied ``provider:model`` list), invokes the
v3 evaluation, and writes the aggregated results JSON to disk.
"""

import json
import sys
import time
from pathlib import Path

# Ensure scripts/ on path so the existing script logic is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Default model roster used when --models all is passed
ALL_MODELS = [
    ("openai", "gpt-4o"),
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-opus-4-5"),
    ("anthropic", "claude-sonnet-4-6"),
    ("anthropic", "claude-haiku-4-5"),
    ("deepseek", "deepseek-chat"),
    ("google", "gemini-2.5-flash"),
    ("google", "gemini-2.5-pro"),
]


def run_evaluation(args) -> None:
    """Run full FormInv evaluation via the v3 runner and write results to disk.

    Resolves the model roster (the built-in :data:`ALL_MODELS` set when
    ``args.models == "all"``, otherwise a comma-separated list of
    ``provider:model`` pairs), runs every model over the dataset, and writes the
    aggregated results JSON to ``args.out``.

    Args:
        args: Parsed CLI arguments with ``dataset``, ``models``, ``out``, and
            ``no_cache`` attributes.

    Returns:
        None.

    Raises:
        ValueError: If ``args.models`` is neither "all" nor a valid list of
            ``provider:model`` pairs.
    """
    from scripts.run_eval_v3 import run_all_models

    if args.models == "all":
        providers = [p for p, _ in ALL_MODELS]
        models = [m for _, m in ALL_MODELS]
    else:
        pairs = [m.strip().split(":", 1) for m in args.models.split(",")]
        if any(len(p) != 2 for p in pairs):
            raise ValueError(
                "--models must be 'all' or a comma-separated list of provider:model pairs, "
                "e.g. openai:gpt-4o,anthropic:claude-sonnet-4-6"
            )
        providers = [p[0] for p in pairs]
        models = [p[1] for p in pairs]

    print("FormInv evaluation")
    print(f"  Dataset : {args.dataset}")
    print(f"  Models  : {list(zip(providers, models, strict=False))}")
    print(f"  Output  : {args.out}")

    t0 = time.time()
    results = run_all_models(
        args.dataset,
        providers,
        models,
        use_cache=not args.no_cache,
    )
    elapsed = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\nCompleted in {elapsed:.0f}s -- results written to {args.out}")
