"""FormInv dataset builder -- delegates to ``scripts/build_dataset_v3.py`` logic.

CLI-facing wrapper that invokes the v3 dataset builder to produce a paraphrase
dataset from the curated theorem set (or a supplied theorem file).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def build_dataset(args) -> None:
    """Build a FormInv paraphrase dataset via the v3 builder.

    Args:
        args: Parsed CLI arguments with ``theorems`` (optional input theorem
            path), ``n`` (number of theorems), and ``out`` (output dataset path)
            attributes.

    Returns:
        None.
    """
    from scripts.build_dataset_v3 import build_dataset as _build

    print("FormInv dataset build")
    print(f"  Theorems : {args.theorems or '(built-in curated set)'}")
    print(f"  N        : {args.n}")
    print(f"  Output   : {args.out}")

    _build(
        n_theorems=args.n,
        out_path=args.out,
    )
