"""FormInv analyze -- delegates to ``scripts/analyze_v3.py`` logic.

CLI-facing wrapper that runs the v3 analysis over an eval results file and
writes a minimal markdown summary pointing back at the source results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def analyze_results(args) -> None:
    """Run FormInv analysis and write a markdown summary.

    Invokes the v3 analysis over ``args.results`` (which prints its report) and
    writes a minimal markdown summary file to ``args.out``.

    Args:
        args: Parsed CLI arguments with ``results`` (input results path) and
            ``out`` (markdown summary output path) attributes.

    Returns:
        None.
    """
    from scripts.analyze_v3 import analyze

    analyze(args.results)

    # If the caller wants a markdown summary file, produce a minimal one.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"# FormInv Analysis\n\nGenerated from: {args.results}\n\n"
        "Re-run `forminv analyze` to regenerate this summary.\n"
    )
    print(f"\nSummary written to {args.out}")
