"""Unified CLI entry point for FormInv.

FormInv measures the *semantic invariance* of LLM mathematical reasoning: whether
a model's verdict on a claim stays constant under meaning-preserving rewording.

Usage:
    forminv eval --dataset data/generated/forminv_v3_50.jsonl --models all
    forminv eval --dataset data/generated/forminv_v3_50.jsonl --models openai:gpt-4o,anthropic:claude-sonnet-4-6
    forminv build --out data/generated/forminv_v3_50.jsonl --n 50
    forminv analyze --results artifacts/results.json
    forminv audit --results artifacts/results.json --threshold 6
    forminv selector --families unpack order --top-k 3
    forminv cir --models all                         # difficulty-controlled CIR (single sample)
    forminv cir --multisample --n 8 --workers 8      # systematic-vs-stochastic CIR
    forminv hf-export --input data/generated/forminv_v3_103.jsonl --out hf_build
    forminv reproduce cir                            # rerun a named paper result

Provider keys (each provider uses its own key): OPENAI_API_KEY, ANTHROPIC_API_KEY,
DEEPSEEK_API_KEY, GOOGLE_API_KEY.
"""

import argparse
import os
import subprocess
import sys

from forminv import __version__

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_script(rel_path: str, script_args: list[str]) -> int:
    """Run a repo script with the current interpreter from the project root.

    Args:
        rel_path: Script path relative to the project root (e.g. ``scripts/x.py``).
        script_args: Extra command-line arguments forwarded to the script.

    Returns:
        The script's integer exit code.
    """
    cmd = [sys.executable, rel_path, *script_args]
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #
def cmd_eval(args: argparse.Namespace) -> None:
    """Run the multi-provider evaluation over a dataset."""
    from forminv.eval.runner import run_evaluation

    run_evaluation(args)


def cmd_build(args: argparse.Namespace) -> None:
    """Build a paraphrase dataset (JSONL)."""
    from forminv.generators.build import build_dataset

    build_dataset(args)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze an evaluation results file and write a markdown summary."""
    from forminv.eval.analyze import analyze_results

    analyze_results(args)


def cmd_audit(args: argparse.Namespace) -> None:
    """Flag low-quality paraphrases via the cross-model consistency audit."""
    from forminv.eval.audit import run_audit

    run_audit(args)


def cmd_selector(args: argparse.Namespace) -> None:
    """Recommend the best model for a target reasoning task (by paraphrase family)."""
    from pathlib import Path

    if args.results is None:
        candidates = [
            Path("artifacts/all_models_results_v2.json"),
            Path("artifacts/results.json"),
        ]
        results_path = next((p for p in candidates if p.exists()), None)
        if results_path is None:
            print(
                "Error: no results file found. Run `forminv eval` first, or pass --results.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        results_path = Path(args.results)

    from forminv.selector import FormInvSelector

    selector = FormInvSelector.from_results(str(results_path))
    recommendations = selector.recommend(families=args.families, top_k=args.top_k)
    print(f"FormInvSelector -- target families: {args.families}")
    print("Expected failure rate (lower = better):")
    for model_key, failure_rate in recommendations:
        print(f"  {model_key:<40} {failure_rate:.1%}")


def cmd_cir(args: argparse.Namespace) -> None:
    """Compute the Conditional Inconsistency Rate.

    Without ``--multisample`` this runs the single-sample, audit-clean CIR on the
    all-models-correct subset (``scripts/cir_rigorous.py``). With ``--multisample``
    it runs the multi-sample (n samples at temperature T) variant across the full
    provider panel, separating systematic from stochastic flips.
    """
    if args.multisample:
        script = "scripts/multisample_cir_multimodel.py"
        passthrough = ["--n", str(args.n), "--workers", str(args.workers)]
        if args.models != "all":
            passthrough += ["--models", args.models]
        if args.limit is not None:
            passthrough += ["--limit", str(args.limit)]
        if args.out:
            passthrough += ["--out", args.out]
    else:
        # cir_rigorous.py reads the response cache directly and takes no arguments.
        script = "scripts/cir_rigorous.py"
        passthrough = []
    sys.exit(_run_script(script, passthrough))


def cmd_hf_export(args: argparse.Namespace) -> None:
    """Convert a FormInv JSONL into the Hugging Face Parquet/config layout."""
    sys.exit(_run_script("scripts/build_hf_dataset.py", ["--input", args.input, "--out", args.out]))


def cmd_reproduce(args: argparse.Namespace) -> None:
    """Rerun a named paper result end-to-end."""
    targets = {
        "decomposition": "scripts/scr_decomposition.py",  # Sec. 6.2 mechanical floor
        "cir": "scripts/cir_rigorous.py",  # Sec. 6.3 CIR (single sample)
        "multisample": "scripts/multisample_cir_multimodel.py",  # Sec. 6.3 multi-sample CIR
        "balanced": "scripts/balanced_eval.py",  # Sec. 5 balanced TRUE/FALSE
        "audit": "scripts/leave_one_out_audit.py",  # paraphrase-quality audit
        "figures": "scripts/generate_figures.py",  # all figures
    }
    script = targets[args.target]
    print(f"Reproducing '{args.target}' -> {script}")
    sys.exit(_run_script(script, []))


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``forminv`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="forminv",
        description="FormInv: a measurement protocol for semantic invariance in LLM mathematical reasoning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `forminv <command> -h` for command-specific options.",
    )
    parser.add_argument("--version", action="version", version=f"forminv {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # eval
    p = sub.add_parser("eval", help="Run multi-provider evaluation on a dataset")
    p.add_argument("--dataset", required=True, help="Path to a dataset JSONL")
    p.add_argument(
        "--models",
        default="all",
        help="'all' (default 9-model roster) or comma-separated provider:model pairs",
    )
    p.add_argument("--out", default="artifacts/results.json", help="Output results JSON")
    p.add_argument("--checkpoint", default="artifacts/checkpoint.json", help="Checkpoint path")
    p.add_argument("--no-cache", action="store_true", help="Disable the response cache")
    p.set_defaults(func=cmd_eval)

    # build
    p = sub.add_parser("build", help="Build a paraphrase dataset")
    p.add_argument("--theorems", help="Input theorem JSONL (optional)")
    p.add_argument("--out", required=True, help="Output dataset JSONL")
    p.add_argument("--n", type=int, default=50, help="Number of theorems (default: 50)")
    p.set_defaults(func=cmd_build)

    # analyze
    p = sub.add_parser("analyze", help="Analyze evaluation results")
    p.add_argument("--results", required=True, help="Results JSON to analyze")
    p.add_argument("--out", default="artifacts/analysis.md", help="Markdown summary output")
    p.set_defaults(func=cmd_analyze)

    # audit
    p = sub.add_parser("audit", help="Paraphrase-quality audit")
    p.add_argument("--results", required=True, help="Results JSON to audit")
    p.add_argument("--threshold", type=int, default=6, help="Min models failing to flag an item (default: 6)")
    p.set_defaults(func=cmd_audit)

    # selector
    p = sub.add_parser("selector", help="Recommend the best model for target paraphrase families")
    p.add_argument("--families", nargs="+", required=True, help="Target families, e.g. --families unpack order")
    p.add_argument("--results", default=None, help="Results JSON (default: auto-detect under artifacts/)")
    p.add_argument("--top-k", type=int, default=3, help="Number of models to show")
    p.set_defaults(func=cmd_selector)

    # cir
    p = sub.add_parser("cir", help="Compute the Conditional Inconsistency Rate")
    p.add_argument(
        "--multisample",
        action="store_true",
        help="Multi-sample CIR (systematic vs. stochastic flips)",
    )
    p.add_argument("--models", default="all", help="'all' or comma-separated model names")
    p.add_argument("--n", type=int, default=8, help="Samples per item (multisample; default: 8)")
    p.add_argument("--workers", type=int, default=8, help="Concurrent API calls (multisample)")
    p.add_argument("--limit", type=int, default=None, help="Cap #theorems (pilot)")
    p.add_argument("--out", default=None, help="Output JSON artifact")
    p.set_defaults(func=cmd_cir)

    # hf-export
    p = sub.add_parser("hf-export", help="Export a dataset to the Hugging Face layout")
    p.add_argument("--input", required=True, help="Input dataset JSONL")
    p.add_argument("--out", required=True, help="Output directory (Parquet configs)")
    p.set_defaults(func=cmd_hf_export)

    # reproduce
    p = sub.add_parser("reproduce", help="Rerun a named paper result")
    p.add_argument(
        "target",
        choices=["decomposition", "cir", "multisample", "balanced", "audit", "figures"],
        help="Which result to reproduce",
    )
    p.set_defaults(func=cmd_reproduce)

    return parser


def main() -> None:
    """Parse arguments and dispatch to the selected subcommand.

    Raises:
        SystemExit: If no subcommand is given (prints help and exits non-zero) or
            a wrapped script returns a non-zero exit code.
    """
    parser = _build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
