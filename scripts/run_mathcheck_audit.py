"""
FormInv cross-model unanimity audit on real MathCheck-GSM data.

arXiv:2407.08733 -- "Is Your Model Really A Good Math Reasoner?"
Dataset: https://huggingface.co/datasets/PremiLab-Math/MathCheck

Experiment design
-----------------
MathCheck organises problems into groups.  Each group has:
  - seed_question            : the original GSM problem
  - problem_understanding_question : a paraphrase of the same problem

For task_type='answerable_judging', both variants have answer='Answerable'
(or 'Unanswerable').  We use *Answerable* pairs so the ground truth is
TRUE for both canonical and paraphrase -- they are semantically equivalent.

FormInv claim under test
------------------------
Cross-model unanimity (>=6/9 models failing a paraphrase while passing
canonical) detects semantically-broken paraphrases in EXTERNAL published
benchmarks.

For a valid paraphrase, neither criterion should fire:
  - almost all models pass canonical  (canonical_pass_rate ~ 1)
  - almost all models pass paraphrase (paraphrase_pass_rate ~ 1)
  - flag does NOT fire

If MathCheck contains a semantically-broken paraphrase (a problem_understanding
variant that actually changes the problem's answerability), the flag WILL fire.

Prompt template (identical for canonical and paraphrase)
--------------------------------------------------------
"Is the following math problem answerable as stated?

Problem: {question}

Answer TRUE if the problem provides enough information to solve it, FALSE if
key information is missing or the problem is self-contradictory."

Expected answer: TRUE (all selected pairs have ground_truth = 'Answerable').

Usage
-----
    cd /Users/noel.thomas/chaos-logic-bench/extensions/forminv
    python scripts/run_mathcheck_audit.py
    python scripts/run_mathcheck_audit.py --n 30          # sample size
    python scripts/run_mathcheck_audit.py --mock           # no API calls
    python scripts/run_mathcheck_audit.py --no-cache       # bypass cache
"""

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from forminv.eval.providers import LLMResponse, call_model

# -- Constants ----------------------------------------------------------------

MATHCHECK_GSM_URL = "https://huggingface.co/datasets/PremiLab-Math/MathCheck/resolve/main/MathCheck-GSM.json"

ANSWERABLE_PROMPT_TEMPLATE = (
    "Is the following math problem answerable as stated?\n\n"
    "Problem: {question}\n\n"
    "Answer TRUE if the problem provides enough information to solve it, "
    "FALSE if key information is missing or the problem is self-contradictory."
)

EXPECTED_GT = "TRUE"  # all 'Answerable' pairs -> TRUE

ALL_MODELS = [
    ("openai", "gpt-4o"),
    ("openai", "gpt-4o-mini"),
    ("openai", "o4-mini"),
    ("anthropic", "claude-sonnet-4-6"),
    ("anthropic", "claude-haiku-4-5"),
    ("deepseek", "deepseek-chat"),
    ("deepseek", "deepseek-reasoner"),
    ("gemini", "gemini-2.5-flash"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
]

UNANIMITY_THRESHOLD = 6  # >=6/9 models


# -- Data loading -------------------------------------------------------------


def load_mathcheck_pairs(n: int | None = None) -> list[dict]:
    """
    Download MathCheck-GSM.json and extract canonical/paraphrase pairs.

    Returns list of dicts:
        group_id, canonical_q, paraphrase_q, expected_gt
    """
    print("Downloading MathCheck-GSM from HuggingFace...", flush=True)
    with urllib.request.urlopen(MATHCHECK_GSM_URL) as resp:
        data = json.loads(resp.read().decode())
    print(f"  Downloaded {len(data)} records.", flush=True)

    groups: dict[str, list] = defaultdict(list)
    for item in data:
        groups[item["group_id"]].append(item)

    pairs = []
    for gid, items in sorted(groups.items(), key=lambda x: int(x[0])):
        aj = [i for i in items if i["task_type"] == "answerable_judging"]
        seed_ans = [i for i in aj if i["question_type"] == "seed_question" and i["answer"] == "Answerable"]
        para_ans = [
            i for i in aj if i["question_type"] == "problem_understanding_question" and i["answer"] == "Answerable"
        ]
        if seed_ans and para_ans:
            pairs.append(
                {
                    "group_id": gid,
                    "canonical_q": seed_ans[0]["question"],
                    "paraphrase_q": para_ans[0]["question"],
                    "expected_gt": EXPECTED_GT,
                }
            )

    print(f"  Extracted {len(pairs)} canonical/paraphrase pairs.")

    if n is not None:
        # Evenly spaced sample so we span the full distribution
        step = max(1, len(pairs) // n)
        pairs = pairs[::step][:n]
        print(f"  Sampled {len(pairs)} pairs (step={step}).")

    return pairs


# -- Model evaluation ---------------------------------------------------------


def query_model(question: str, provider: str, model: str, use_cache: bool = True) -> dict:
    """Wrap a question in the answerable prompt and call the model."""
    prompt = ANSWERABLE_PROMPT_TEMPLATE.format(question=question)
    try:
        resp: LLMResponse = call_model(prompt, provider=provider, model=model, use_cache=use_cache)
        return {
            "parsed": resp.parsed,
            "raw": resp.raw[:300],
            "latency_s": resp.latency_s,
            "error": None,
        }
    except Exception as exc:
        return {
            "parsed": None,
            "raw": "",
            "latency_s": 0.0,
            "error": str(exc),
        }


def run_audit(pairs: list[dict], models: list[tuple], use_cache: bool = True) -> dict:
    """
    For each pair, query every model on canonical and paraphrase.
    Returns nested dict: group_id -> {canonical: {model_key: result}, paraphrase: {...}}
    """
    results: dict = {}
    total = len(pairs) * len(models) * 2
    done = 0

    for pair in pairs:
        gid = pair["group_id"]
        results[gid] = {
            "canonical_q": pair["canonical_q"],
            "paraphrase_q": pair["paraphrase_q"],
            "expected_gt": pair["expected_gt"],
            "canonical": {},
            "paraphrase": {},
        }

        for side, q in [("canonical", pair["canonical_q"]), ("paraphrase", pair["paraphrase_q"])]:
            for provider, model in models:
                model_key = f"{provider}/{model.replace('/', '_')}"
                res = query_model(q, provider, model, use_cache=use_cache)
                results[gid][side][model_key] = res
                done += 1
                status = res["parsed"] if res["parsed"] else ("ERR" if res["error"] else "?")
                print(
                    f"  [{done:>4}/{total}] group={gid} {side:>10} {model_key:<50} -> {status}",
                    flush=True,
                )

    return results


# -- Audit statistics ----------------------------------------------------------


def compute_stats(results: dict) -> dict:
    """
    For each group compute:
      - canonical_pass_rate   : fraction of models answering TRUE on canonical
      - paraphrase_pass_rate  : fraction of models answering TRUE on paraphrase
      - paraphrase_fail_count : count of models answering FALSE on paraphrase
      - canonical_pass_count  : count of models answering TRUE on canonical
      - flagged               : paraphrase_fail_count >= threshold AND
                                canonical_pass_count >= threshold
      - flag_reason           : human-readable

    A flag means the paraphrase diverges behaviourally from the canonical
    (despite having the same labeled ground truth in MathCheck).
    """
    stats = {}
    for gid, data in results.items():
        gt = data["expected_gt"]
        model_keys = list(data["canonical"].keys())
        n = len(model_keys)

        canon_pass = sum(1 for mk in model_keys if data["canonical"][mk]["parsed"] == gt)
        para_fail = sum(
            1
            for mk in model_keys
            if data["paraphrase"][mk]["parsed"] is not None and data["paraphrase"][mk]["parsed"] != gt
        )
        # Also count parseable responses to assess reliability
        canon_parseable = sum(1 for mk in model_keys if data["canonical"][mk]["parsed"] is not None)
        para_parseable = sum(1 for mk in model_keys if data["paraphrase"][mk]["parsed"] is not None)

        flagged = para_fail >= UNANIMITY_THRESHOLD and canon_pass >= UNANIMITY_THRESHOLD

        stats[gid] = {
            "n_models": n,
            "canonical_pass_count": canon_pass,
            "canonical_pass_rate": canon_pass / max(n, 1),
            "paraphrase_fail_count": para_fail,
            "paraphrase_fail_rate": para_fail / max(n, 1),
            "canon_parseable": canon_parseable,
            "para_parseable": para_parseable,
            "flagged": flagged,
        }

    return stats


# -- Reporting -----------------------------------------------------------------


def print_summary(pairs: list[dict], stats: dict, models: list[tuple]) -> None:
    n_models = len(models)
    n_items = len(stats)
    flagged_ids = [gid for gid, s in stats.items() if s["flagged"]]

    print(f"\n{'=' * 80}")
    print("FormInv x MathCheck-GSM Cross-Model Unanimity Audit")
    print(f"Models: {n_models}  |  Items: {n_items}  |  Threshold: {UNANIMITY_THRESHOLD}/{n_models}")
    print(f"{'=' * 80}")
    print(f"{'GID':<6} {'Canon+':>7} {'Para-':>7} {'Flagged':>8}  {'Canonical Q (first 60 chars)'}")
    print("-" * 100)

    for pair in pairs:
        gid = pair["group_id"]
        s = stats.get(gid, {})
        flag = "FLAG" if s.get("flagged") else ""
        print(
            f"{gid:<6} {s.get('canonical_pass_count', 0):>3}/{s.get('n_models', 0):<3} "
            f"{s.get('paraphrase_fail_count', 0):>3}/{s.get('n_models', 0):<3} "
            f"{flag:>8}  "
            f"{pair['canonical_q'][:60]}"
        )

    print(f"\n{'=' * 80}")
    print(f"Flagged items: {len(flagged_ids)}/{n_items}")

    if flagged_ids:
        print("\nFlagged groups (candidate bad paraphrases in MathCheck):")
        for gid in flagged_ids:
            s = stats[gid]
            pair = next(p for p in pairs if p["group_id"] == gid)
            print(f"\n  Group {gid}:")
            print(f"    Canonical  : {pair['canonical_q'][:120]}")
            print(f"    Paraphrase : {pair['paraphrase_q'][:120]}")
            print(f"    Models passing canonical : {s['canonical_pass_count']}/{s['n_models']}")
            print(f"    Models failing paraphrase: {s['paraphrase_fail_count']}/{s['n_models']}")
    else:
        print("\nNo items flagged -- MathCheck paraphrases behave consistently across models.")
        print("This is the precision/control result: audit does not fire on curated pairs.")


def write_markdown_report(pairs: list[dict], stats: dict, results: dict, models: list[tuple], out_path: Path) -> None:
    flagged_ids = [gid for gid, s in stats.items() if s["flagged"]]
    n_flagged = len(flagged_ids)
    n_items = len(stats)
    n_models = len(models)

    lines = [
        "# FormInv Cross-Model Unanimity Audit on MathCheck-GSM",
        "",
        "**Source**: MathCheck-GSM (arXiv:2407.08733) -- "
        "[PremiLab-Math/MathCheck](https://huggingface.co/datasets/PremiLab-Math/MathCheck)",
        "**Date**: 2026-05-26",
        f"**Items**: {n_items} canonical/paraphrase pairs (answerable_judging, Answerable)",
        f"**Models**: {n_models} (OpenAI, Anthropic, DeepSeek, Gemini, Llama via OpenRouter)",
        f"**Unanimity threshold**: >={UNANIMITY_THRESHOLD}/{n_models} models failing paraphrase "
        f"while >={UNANIMITY_THRESHOLD}/{n_models} pass canonical",
        "",
        "## What This Tests",
        "",
        "MathCheck groups each GSM problem into four variant types. The `seed_question`"
        " is the original; `problem_understanding_question` is a surface-level paraphrase"
        " with the same labeled answer (`Answerable`). Both variants should elicit"
        " identical model behaviour if the paraphrase is semantically equivalent.",
        "",
        "FormInv's cross-model unanimity audit fires when >=6/9 models fail the"
        " paraphrase while >=6/9 pass the canonical -- signalling that the paraphrase"
        " has changed the underlying question despite carrying the same label.",
        "",
        "## Prompt Template",
        "",
        "```",
        ANSWERABLE_PROMPT_TEMPLATE.replace("{question}", "<problem text>"),
        "```",
        "",
        "**Expected answer**: TRUE (all pairs have MathCheck label = Answerable)",
        "",
        "## Per-Item Results",
        "",
        "| Group | Canon+ | Para- | Flagged | Canonical (first 80 chars) |",
        "|-------|--------|-------|---------|---------------------------|",
    ]

    for pair in pairs:
        gid = pair["group_id"]
        s = stats.get(gid, {})
        flag = "**FLAG**" if s.get("flagged") else ""
        canon_short = pair["canonical_q"][:80].replace("|", "\\|")
        lines.append(
            f"| {gid} | {s.get('canonical_pass_count', 0)}/{s.get('n_models', 0)} "
            f"| {s.get('paraphrase_fail_count', 0)}/{s.get('n_models', 0)} "
            f"| {flag} | {canon_short} |"
        )

    lines += [
        "",
        f"## Flagged Items: {n_flagged}/{n_items}",
        "",
    ]

    if flagged_ids:
        for gid in flagged_ids:
            s = stats[gid]
            pair = next(p for p in pairs if p["group_id"] == gid)
            lines += [
                f"### Group {gid}",
                "",
                f"**Canonical**: {pair['canonical_q']}",
                "",
                f"**Paraphrase**: {pair['paraphrase_q']}",
                "",
                f"- Models passing canonical: {s['canonical_pass_count']}/{s['n_models']}",
                f"- Models failing paraphrase: {s['paraphrase_fail_count']}/{s['n_models']}",
                "",
                "| Model | Canonical | Paraphrase |",
                "|-------|-----------|------------|",
            ]
            for mk in results[gid]["canonical"]:
                cp = results[gid]["canonical"][mk].get("parsed", "?")
                pp = results[gid]["paraphrase"][mk].get("parsed", "?")
                lines.append(f"| {mk} | {cp} | {pp} |")
            lines.append("")
    else:
        lines += [
            "No items were flagged.",
            "",
            "This is the **precision/control result**: FormInv's audit does not fire",
            "on a curated external benchmark where paraphrases are genuinely semantically",
            "equivalent, confirming the audit has low false-positive rate.",
            "",
        ]

    # Conclusion
    if n_flagged > 0:
        conclusion = (
            f"FormInv's cross-model unanimity audit detected **{n_flagged} candidate"
            f" bad paraphrase(s)** in MathCheck-GSM (out of {n_items} tested). "
            f"These items show high canonical pass rates but >={UNANIMITY_THRESHOLD}/{n_models}"
            f" models failing the paraphrase, indicating the paraphrase may alter the"
            f" problem's semantic content despite carrying the same label. "
            f"Manual inspection of the flagged pairs is required to confirm whether"
            f" the divergence reflects a genuine labeling error or a difficulty artifact."
        )
    else:
        conclusion = (
            f"FormInv's cross-model unanimity audit found **0 flagged items** across"
            f" {n_items} MathCheck-GSM answerable_judging pairs. This establishes that"
            f" the audit has low false-positive rate on a curated external benchmark:"
            f" when paraphrases are semantically equivalent, the unanimity signal does"
            f" not fire. Combined with FormInv's true-positive results on known-bad"
            f" paraphrases, this supports the audit's specificity."
        )

    lines += [
        "## Conclusion",
        "",
        conclusion,
        "",
        "---",
        "*Generated by FormInv scripts/run_mathcheck_audit.py -- "
        "arXiv:2407.08733 x FormInv cross-model unanimity audit*",
    ]

    out_path.write_text("\n".join(lines))
    print(f"\nMarkdown report saved to: {out_path}")


# -- Main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="FormInv cross-model unanimity audit on real MathCheck-GSM data")
    parser.add_argument("--n", type=int, default=30, help="Number of pairs to audit (default 30; use 0 for all 129)")
    parser.add_argument("--mock", action="store_true", help="Use mock provider -- no API calls, all responses TRUE")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the response cache")
    parser.add_argument(
        "--raw-out",
        default="artifacts/mathcheck_audit_raw.json",
        help="Where to save raw per-item results (JSON)",
    )
    parser.add_argument(
        "--report-out",
        default="artifacts/mathcheck_audit_results.md",
        help="Where to save the markdown report",
    )
    args = parser.parse_args()

    n = args.n if args.n > 0 else None

    if args.mock:
        models = [("mock", "mock-always-true")]
    else:
        models = ALL_MODELS

    pairs = load_mathcheck_pairs(n=n)

    print("\nRunning audit:")
    print(f"  Pairs   : {len(pairs)}")
    print(f"  Models  : {len(models)}")
    print("  Sides   : canonical + paraphrase")
    print(f"  Total   : {len(pairs) * len(models) * 2} API calls (minus cache hits)")
    print()

    results = run_audit(pairs, models, use_cache=not args.no_cache)
    stats = compute_stats(results)
    print_summary(pairs, stats, models)

    # Save raw JSON
    raw_path = Path(args.raw_out)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({"pairs": pairs, "results": results, "stats": stats}, indent=2))
    print(f"Raw results saved to: {args.raw_out}")

    # Save markdown report
    report_path = Path(args.report_out)
    write_markdown_report(pairs, stats, results, models, report_path)

    return results, stats


if __name__ == "__main__":
    main()
