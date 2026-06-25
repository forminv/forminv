"""
FormInv x MathCheck-GSM Full 129-Group Ranking-Change Experiment

arXiv:2407.08733 -- "Is Your Model Really A Good Math Reasoner?"
Dataset: https://huggingface.co/datasets/PremiLab-Math/MathCheck

Extends the 30-group pilot to all 129 groups.
Reuses all existing cache from the 30-group run.

Outputs
-------
  data/generated/mathcheck_formInv_full_129.jsonl  (258 items)
  artifacts/mathcheck_129_results.json
  artifacts/mathcheck_129_ranking_report.md

Usage
-----
    cd /path/to/forminv
    python scripts/run_mathcheck_ranking_129.py               # live API calls
    python scripts/run_mathcheck_ranking_129.py --mock        # no API calls
    python scripts/run_mathcheck_ranking_129.py --n 32        # smoke test
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Re-use all the evaluation machinery from the 30-group script
from scripts.run_mathcheck_ranking import (
    FORMINV_DETECT_THRESHOLD,
    MODELS,
    compute_metrics,
    detect_bad_paraphrases,
    extract_pairs,
    rank_models,
    run_eval,
)

# -- Pairwise ranking analysis -------------------------------------------------


def pairwise_ranking_changes(ranks_with: list, ranks_without: list):
    """
    Compare pairwise orderings WITH vs WITHOUT bad paraphrases.

    Returns three lists:
      full_reversals   -- (A, B, scr_with_A, scr_with_B, scr_without_A, scr_without_B)
                         where sign flips (A > B WITH but B > A WITHOUT, both non-tied)
      tie_collapses    -- same tuple, where one condition is a tie, the other is not
      no_change        -- same tuple, no rank change
    """
    scr_with = {mk: v for mk, v in ranks_with}
    scr_without = {mk: v for mk, v in ranks_without}
    model_keys = [mk for mk, _ in ranks_with]

    full_reversals = []
    tie_collapses = []
    no_change = []

    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            A, B = model_keys[i], model_keys[j]
            delta_with = scr_with[A] - scr_with[B]
            delta_without = scr_without[A] - scr_without[B]

            # Use 1e-6 tolerance to handle float arithmetic
            sign_with = (delta_with > 1e-6) - (delta_with < -1e-6)
            sign_without = (delta_without > 1e-6) - (delta_without < -1e-6)

            entry = (A, B, scr_with[A], scr_with[B], scr_without[A], scr_without[B])

            if sign_with != 0 and sign_without != 0 and sign_with != sign_without:
                full_reversals.append(entry)
            elif sign_with != sign_without:
                # One is a tie (0) and the other is not
                tie_collapses.append(entry)
            else:
                no_change.append(entry)

    return full_reversals, tie_collapses, no_change


# -- JSONL dump ----------------------------------------------------------------


def dump_jsonl(results: dict, out_path: Path) -> None:
    """Write 258 items (129 canonical + 129 PU) in the existing schema format."""
    records = []
    for gid in sorted(results.keys(), key=lambda x: int(x)):
        data = results[gid]
        gid_int = int(gid)
        gid_str = f"{gid_int:02d}"

        for variant_key, family, variant_label in [
            ("canonical", "canonical", "seed_question"),
            ("pu", "variation", "problem_understanding_question"),
        ]:
            q = data[f"{variant_key}_q"]
            records.append(
                {
                    "id": f"mathcheck_g{gid_str}_{variant_key}",
                    "theorem_id": f"mathcheck_g{gid_str}",
                    "family": family,
                    "nl_question": q,
                    "ground_truth": "TRUE",
                    "canonical_nl": data["canonical_q"],
                    "expected_answer": data["expected_answer"],
                    "group_id": gid,
                    "variant": variant_label,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"JSONL written: {out_path} ({len(records)} items)")


# -- Report builder ------------------------------------------------------------


def build_full_report(results: dict, models: list[tuple]) -> str:
    n_models = len(models)
    model_keys = [f"{p}/{m}" for p, m in models]
    n_groups = len(results)

    # Detect bad paraphrases via FormInv rule
    detected_bad = detect_bad_paraphrases(results, models, threshold=FORMINV_DETECT_THRESHOLD)
    n_bad = len(detected_bad)

    # Compute metrics
    metrics_all = compute_metrics(results, models, exclude_groups=None)
    metrics_clean = compute_metrics(results, models, exclude_groups=detected_bad)

    ranks_all = rank_models(metrics_all, "SCR")
    ranks_clean = rank_models(metrics_clean, "SCR")

    full_reversals, tie_collapses, no_change = pairwise_ranking_changes(ranks_all, ranks_clean)

    has_full_reversal = len(full_reversals) > 0

    def fmt_rank_block(ranks: list, metrics: dict, label: str) -> list[str]:
        lines = [f"Model Rankings {label}:"]
        for rank_idx, (mk, scr) in enumerate(ranks, start=1):
            short = mk.split("/")[-1]
            lines.append(f"  {rank_idx}. {short:<35} SCR={scr:.1%}")
        return lines

    def fmt_pair(A, B, wa, wb, woa, wob) -> str:
        a_short = A.split("/")[-1]
        b_short = B.split("/")[-1]
        return (
            f"  {a_short} vs {b_short}: "
            f"WITH {wa:.1%}/{wb:.1%} ({'+' if wa > wb else ''}{wa - wb:.1%}), "
            f"WITHOUT {woa:.1%}/{wob:.1%} ({'+' if woa > wob else ''}{woa - wob:.1%})"
        )

    # Per-group table for the report
    def fmt_per_group_table() -> list[str]:
        lines = ["Per-Group Results (C=canonical correct, P=PU correct; T=pass F=fail)", ""]
        header = f"{'Grp':>4}  {'ExpAns':>8}"
        for provider, model in models:
            short = model.split("-")[-1][:6]
            header += f"  C/{short}"
        header += "  Flag"
        lines.append(header)
        lines.append("-" * (len(header) + 10))

        for gid in sorted(results.keys(), key=lambda x: int(x)):
            data = results[gid]
            row = f"{gid:>4}  {data['expected_answer']:>8.1f}"
            canon_pass = 0
            pu_fail = 0
            for provider, model in models:
                mk = f"{provider}/{model}"
                c = data["canonical"].get(mk, {}).get("correct", False)
                p = data["pu"].get(mk, {}).get("correct", False)
                row += f"  {'T' if c else 'F'}/{'T' if p else 'F'}"
                if c:
                    canon_pass += 1
                if not p:
                    pu_fail += 1
            flag = ""
            if gid in detected_bad:
                flag = "  <- FORMINV_FLAGGED"
            lines.append(row + flag)
        return lines

    # -- Assemble report -------------------------------------------------------
    lines = [
        "MathCheck Full 129-Group Experiment",
        "====================================",
        f"Total groups: {n_groups}",
        f"Bad paraphrases flagged by FormInv unanimity: {n_bad}",
        f"Error rate: {n_bad}/{n_groups} = {n_bad / n_groups:.1%}",
        "",
        f"FormInv rule: canon_pass >= {FORMINV_DETECT_THRESHOLD}/{n_models} AND "
        f"pu_fail >= {FORMINV_DETECT_THRESHOLD}/{n_models}",
        f"Flagged groups: {', '.join(sorted(detected_bad, key=int)) if detected_bad else 'none'}",
        "",
    ]

    lines += fmt_rank_block(ranks_all, metrics_all, "WITH bad paraphrases:")
    lines += [""]
    lines += fmt_rank_block(ranks_clean, metrics_clean, "WITHOUT bad paraphrases:")
    lines += [""]

    lines.append("Ranking changes:")

    if full_reversals:
        lines.append(f"  Full reversals ({len(full_reversals)}):")
        for entry in full_reversals:
            lines.append(fmt_pair(*entry))
    else:
        lines.append("  Full reversals: none")

    if tie_collapses:
        lines.append(f"  Tie-collapses ({len(tie_collapses)}):")
        for entry in tie_collapses:
            lines.append(fmt_pair(*entry))
    else:
        lines.append("  Tie-collapses: none")

    if no_change:
        lines.append(f"  No change ({len(no_change)}):")
        for entry in no_change:
            lines.append(fmt_pair(*entry))

    lines += [
        "",
        f"CONCLUSION: Does removing bad paraphrases cause any full model ranking reversal? "
        f"{'YES' if has_full_reversal else 'NO'}",
        "",
    ]

    # Pre-specified group 38 audit
    lines.append("Pre-specified group audit:")
    for gid in sorted(["25", "26", "27", "38"], key=int):
        if gid in results:
            status = "FLAGGED" if gid in detected_bad else "not flagged"
            lines.append(f"  Group {gid}: {status}")
        else:
            lines.append(f"  Group {gid}: NOT IN RESULTS (unexpected)")
    lines.append("")

    # Full metrics table
    lines.append("Full SCR Metrics Table:")
    lines.append(f"{'Model':<40} {'Acc(canon)':>12} {'Acc(PU)':>10} {'SCR(all)':>10} {'SCR(clean)':>12}")
    lines.append("-" * 88)
    for mk in model_keys:
        ma = metrics_all[mk]
        mc = metrics_clean[mk]
        lines.append(
            f"{mk:<40} {ma['acc_canonical']:>11.1%} {ma['acc_pu']:>10.1%} {ma['SCR']:>10.1%} {mc['SCR']:>12.1%}"
        )
    lines.append("")

    lines += ["--- Per-group detail ---", ""]
    lines += fmt_per_group_table()

    return "\n".join(lines)


# -- Main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="FormInv x MathCheck full 129-group ranking-change experiment")
    parser.add_argument("--mock", action="store_true", help="Mock provider (no API calls)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass response cache")
    parser.add_argument("--n", type=int, default=129, help="Number of groups to evaluate (default 129)")
    parser.add_argument("--jsonl-out", default="data/generated/mathcheck_formInv_full_129.jsonl")
    parser.add_argument("--raw-out", default="artifacts/mathcheck_129_results.json")
    parser.add_argument("--report-out", default="artifacts/mathcheck_129_ranking_report.md")
    args = parser.parse_args()

    pairs = extract_pairs(n_groups=args.n)

    if args.mock:
        models = [("mock", "mock-model")]
    else:
        models = MODELS

    n_groups = len(pairs)
    total_calls = n_groups * len(models) * 2
    print("\nRunning evaluation:")
    print(f"  Groups  : {n_groups}")
    print(f"  Models  : {len(models)} -- {[m for _, m in models]}")
    print(f"  Total   : {total_calls} API calls (minus cache hits)")
    print()

    results = run_eval(pairs, models, use_cache=not args.no_cache)

    # Save raw JSON
    raw_path = Path(args.raw_out)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(results, indent=2))
    print(f"\nRaw results saved to: {args.raw_out}")

    # Save JSONL
    jsonl_path = Path(args.jsonl_out)
    dump_jsonl(results, jsonl_path)

    # Build and save report
    report = build_full_report(results, models)
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"Report saved to: {args.report_out}")

    # Print summary to console
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
