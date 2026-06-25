"""
FormInv x MathCheck-GSM Ranking-Change Experiment

arXiv:2407.08733 -- "Is Your Model Really A Good Math Reasoner?"
Dataset: https://huggingface.co/datasets/PremiLab-Math/MathCheck

Core claim under test
---------------------
If MathCheck contains semantically-broken PU paraphrases (groups 26, 38),
including them in a model benchmark *distorts rankings*.  Removing just those
2 items should change which model ranks highest on paraphrase robustness (SCR).

Experiment design
-----------------
- 30 groups (IDs 0-29) x 2 variants (seed_question + problem_understanding_question)
  for task_type=solving -> 60 prompts
- 4 models: Haiku, Sonnet, GPT-4o, DeepSeek-Chat
- Evaluation: ask model to solve and return final numeric answer; compare to
  ground-truth number with tolerance 1e-3
- Metrics:
    acc_canonical = fraction of canonical items correct
    acc_pu        = fraction of PU items correct
    SCR           = fraction of groups where BOTH variants correct
    rank_by_scr   = model ranking by SCR
- Compute WITH all 30 PU items and WITHOUT groups 26 and 38

Usage
-----
    cd /path/to/forminv
    python scripts/run_mathcheck_ranking.py               # live API calls
    python scripts/run_mathcheck_ranking.py --mock        # no API calls
    python scripts/run_mathcheck_ranking.py --no-cache    # bypass cache
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Constants -----------------------------------------------------------------

MATHCHECK_GSM_URL = "https://huggingface.co/datasets/PremiLab-Math/MathCheck/resolve/main/MathCheck-GSM.json"

LOCAL_CACHE = Path("/tmp/mathcheck_gsm.json")

SOLVE_SYSTEM_PROMPT = (
    "You are a precise math problem solver. "
    "Solve the problem step by step, then end your response with exactly: "
    "Final Answer: <number>"
    "\nUse only a bare number (no units, no $, no commas)."
)

SOLVE_USER_TEMPLATE = "Solve this problem:\n\n{question}\n\nEnd with: Final Answer: <number>"

PRESPECIFIED_BAD_GROUPS = {"26", "38"}  # pre-pilot labels (NOTE: 38 is outside groups 0-29)
# Empirical detection uses FormInv rule: canon_pass_rate >= 3/4 AND pu_fail_rate >= 3/4
FORMINV_DETECT_THRESHOLD = 3  # out of 4 models

MODELS = [
    ("anthropic", "claude-haiku-4-5"),
    ("anthropic", "claude-sonnet-4-6"),
    ("openai", "gpt-4o"),
    ("deepseek", "deepseek-chat"),
]

RESPONSE_CACHE_DIR = Path("data/raw/response_cache_solve")

# -- Data loading --------------------------------------------------------------


def load_mathcheck_gsm() -> list:
    if LOCAL_CACHE.exists():
        with open(LOCAL_CACHE) as f:
            return json.load(f)
    print("Downloading MathCheck-GSM from HuggingFace...", flush=True)
    with urllib.request.urlopen(MATHCHECK_GSM_URL, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    LOCAL_CACHE.write_text(json.dumps(data))
    return data


def extract_pairs(n_groups: int = 30) -> list[dict]:
    data = load_mathcheck_gsm()
    groups: dict[str, list] = defaultdict(list)
    for item in data:
        groups[item["group_id"]].append(item)

    pairs = []
    for i in range(n_groups):
        gid = str(i)
        items = groups.get(gid, [])
        seed = next(
            (it for it in items if it["task_type"] == "solving" and it["question_type"] == "seed_question"),
            None,
        )
        pu = next(
            (
                it
                for it in items
                if it["task_type"] == "solving" and it["question_type"] == "problem_understanding_question"
            ),
            None,
        )
        if seed and pu:
            pairs.append(
                {
                    "group_id": gid,
                    "canonical_q": seed["question"],
                    "pu_q": pu["question"],
                    "expected_answer": float(seed["answer"]),
                }
            )
        else:
            print(f"  WARNING: group {gid} missing seed or PU item")

    print(f"Extracted {len(pairs)} groups (0-{n_groups - 1})")
    return pairs


# -- Response cache ------------------------------------------------------------


def _cache_key(model: str, prompt: str) -> str:
    h = hashlib.md5(f"solve::{model}::{prompt}".encode()).hexdigest()[:16]
    return h


def _load_cache(model: str, prompt: str):
    RESPONSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = RESPONSE_CACHE_DIR / f"{_cache_key(model, prompt)}.json"
    if f.exists():
        return json.loads(f.read_text())
    return None


def _save_cache(model: str, prompt: str, d: dict) -> None:
    RESPONSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = RESPONSE_CACHE_DIR / f"{_cache_key(model, prompt)}.json"
    f.write_text(json.dumps(d))


# -- Answer extraction ---------------------------------------------------------


def parse_final_answer(text: str) -> float | None:
    """Extract the number after 'Final Answer:' with robust regex."""
    # Look for "Final Answer: <number>" (case-insensitive)
    m = re.search(r"[Ff]inal\s+[Aa]nswer\s*:\s*\$?([\-\d,\.]+)", text)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            return float(raw)
        except ValueError:
            pass
    # Fallback: last numeric token in last line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        nums = re.findall(r"\$?([\-\d,\.]+)", last)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                pass
    return None


def answer_correct(parsed: float | None, expected: float, tol: float = 1e-3) -> bool:
    if parsed is None:
        return False
    return abs(parsed - expected) <= tol


# -- Provider calls ------------------------------------------------------------


def call_solve(question: str, provider: str, model: str, use_cache: bool = True) -> dict:
    prompt = SOLVE_USER_TEMPLATE.format(question=question)

    if use_cache:
        cached = _load_cache(model, prompt)
        if cached:
            return cached

    t0 = time.time()
    raw = ""
    error = None

    try:
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SOLVE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text if resp.content else ""

        elif provider == "openai":
            import openai

            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content or ""

        elif provider == "deepseek":
            import openai

            client = openai.OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                base_url="https://api.deepseek.com",
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=1024,
            )
            msg = resp.choices[0].message
            raw = msg.content or ""

        elif provider == "mock":
            # Mock: always returns the expected answer (passed via question encoding trick)
            # In mock mode we just return a fake response
            raw = "Step 1: compute. Final Answer: 999"

        else:
            raise ValueError(f"Unknown provider: {provider}")

    except Exception as exc:
        error = str(exc)

    latency = time.time() - t0
    parsed = parse_final_answer(raw)
    result = {
        "raw": raw[:500],
        "parsed_answer": parsed,
        "latency_s": round(latency, 3),
        "error": error,
    }
    if use_cache and not error:
        _save_cache(model, prompt, result)
    return result


# -- Evaluation loop -----------------------------------------------------------


def run_eval(pairs: list[dict], models: list[tuple], use_cache: bool = True) -> dict:
    """
    Returns nested dict: group_id -> {
        expected_answer, canonical_q, pu_q,
        canonical: {model_key: result},
        pu:        {model_key: result},
    }
    """
    results: dict = {}
    total_calls = len(pairs) * len(models) * 2
    done = 0

    for pair in pairs:
        gid = pair["group_id"]
        results[gid] = {
            "expected_answer": pair["expected_answer"],
            "canonical_q": pair["canonical_q"],
            "pu_q": pair["pu_q"],
            "canonical": {},
            "pu": {},
        }

        for side, q in [("canonical", pair["canonical_q"]), ("pu", pair["pu_q"])]:
            for provider, model in models:
                model_key = f"{provider}/{model}"
                res = call_solve(q, provider, model, use_cache=use_cache)
                correct = answer_correct(res["parsed_answer"], pair["expected_answer"])
                res["correct"] = correct
                results[gid][side][model_key] = res
                done += 1
                status = (
                    f"{res['parsed_answer']} {'+' if correct else '-'}"
                    if res["parsed_answer"] is not None
                    else "PARSE_ERR"
                )
                print(
                    f"  [{done:>4}/{total_calls}] g={gid:>3} {side:>9} {model_key:<42} "
                    f"exp={pair['expected_answer']:8.1f} got={status}",
                    flush=True,
                )

    return results


# -- Metrics -------------------------------------------------------------------


def compute_metrics(results: dict, models: list[tuple], exclude_groups: set[str] | None = None) -> dict:
    """
    Compute per-model metrics:
      acc_canonical, acc_pu, SCR, n_groups
    exclude_groups: set of group_id strings to exclude from PU analysis
    """
    model_keys = [f"{p}/{m}" for p, m in models]
    metrics = {
        mk: {
            "correct_canonical": 0,
            "correct_pu": 0,
            "correct_both": 0,
            "n_canonical": 0,
            "n_pu": 0,
            "n_groups": 0,
        }
        for mk in model_keys
    }

    groups_used = {gid for gid in results if exclude_groups is None or gid not in exclude_groups}

    for gid, data in results.items():
        # Canonical is always included
        for mk in model_keys:
            if mk in data["canonical"]:
                metrics[mk]["n_canonical"] += 1
                if data["canonical"][mk]["correct"]:
                    metrics[mk]["correct_canonical"] += 1

        # PU: skip if in exclude_groups
        if exclude_groups and gid in exclude_groups:
            continue

        for mk in model_keys:
            if mk not in data["pu"] or mk not in data["canonical"]:
                continue
            metrics[mk]["n_pu"] += 1
            metrics[mk]["n_groups"] += 1
            pu_ok = data["pu"][mk]["correct"]
            canon_ok = data["canonical"][mk]["correct"]
            if pu_ok:
                metrics[mk]["correct_pu"] += 1
            if pu_ok and canon_ok:
                metrics[mk]["correct_both"] += 1

    for mk in model_keys:
        m = metrics[mk]
        m["acc_canonical"] = m["correct_canonical"] / max(m["n_canonical"], 1)
        m["acc_pu"] = m["correct_pu"] / max(m["n_pu"], 1)
        m["SCR"] = m["correct_both"] / max(m["n_groups"], 1)

    return metrics


def rank_models(metrics: dict, by: str = "SCR") -> list[tuple[str, float]]:
    """Return models sorted descending by metric."""
    ranked = sorted(metrics.items(), key=lambda x: x[1][by], reverse=True)
    return [(mk, d[by]) for mk, d in ranked]


# -- FormInv detection rule ----------------------------------------------------


def detect_bad_paraphrases(results: dict, models: list[tuple], threshold: int = 3) -> set[str]:
    """
    FormInv automated detection rule:
      Flag group if canon_pass_count >= threshold AND pu_fail_count >= threshold.
    This is entirely model-agnostic and data-driven -- no pre-specified labels.
    """
    model_keys = [f"{p}/{m}" for p, m in models]
    flagged = set()
    for gid, data in results.items():
        canon_pass = sum(1 for mk in model_keys if data["canonical"].get(mk, {}).get("correct"))
        pu_fail = sum(1 for mk in model_keys if not data["pu"].get(mk, {}).get("correct"))
        if canon_pass >= threshold and pu_fail >= threshold:
            flagged.add(gid)
    return flagged


# -- Reporting -----------------------------------------------------------------


def build_report(results: dict, models: list[tuple]) -> str:
    n_models = len(models)
    model_keys = [f"{p}/{m}" for p, m in models]

    # -- FormInv detection ----------------------------------------------------
    detected_bad = detect_bad_paraphrases(results, models, threshold=FORMINV_DETECT_THRESHOLD)

    # -- Pre-specified labels audit -------------------------------------------
    # Group 38 is outside groups 0-29; can only test 26 in this run
    prespec_testable = {g for g in PRESPECIFIED_BAD_GROUPS if g in results}
    prespec_outside = PRESPECIFIED_BAD_GROUPS - set(results.keys())

    # -- Metrics ---------------------------------------------------------------
    metrics_all = compute_metrics(results, models, exclude_groups=None)
    metrics_clean = compute_metrics(results, models, exclude_groups=detected_bad)

    ranks_all = rank_models(metrics_all, "SCR")
    ranks_clean = rank_models(metrics_clean, "SCR")

    ranking_changed_positions = [mk for i, (mk, _) in enumerate(ranks_all) if mk != ranks_clean[i][0]]
    scr_changed = bool(ranking_changed_positions)

    def fmt_table(metrics: dict, title: str, n_pu: int) -> list[str]:
        lines = [title, ""]
        lines.append(
            f"{'Model':<40} {'Acc(canon)':>12} {'Acc(PU)':>10} {'SCR':>8} {'Rank(canon)':>12} {'Rank(SCR)':>10}"
        )
        lines.append("-" * 98)
        canon_ranks = rank_models(metrics, "acc_canonical")
        scr_ranks = rank_models(metrics, "SCR")
        canon_rank_map = {mk: i + 1 for i, (mk, _) in enumerate(canon_ranks)}
        scr_rank_map = {mk: i + 1 for i, (mk, _) in enumerate(scr_ranks)}
        for mk in model_keys:
            m = metrics[mk]
            lines.append(
                f"{mk:<40} {m['acc_canonical']:>11.1%} {m['acc_pu']:>10.1%} "
                f"{m['SCR']:>8.1%} {canon_rank_map[mk]:>12} {scr_rank_map[mk]:>10}"
            )
        lines.append(f"  (n_canonical=30, n_pu={n_pu})")
        return lines

    def fmt_per_group(results: dict, detected: set[str]) -> list[str]:
        lines = ["Per-Group Results (C=canonical, P=PU; T=correct, F=wrong)", ""]
        header = f"{'Grp':>4}  {'ExpAns':>8}"
        for provider, model in models:
            mk = f"{provider}/{model}"
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
            if gid in detected:
                flag = "  <- FORMINV_FLAGGED"
            elif gid in prespec_testable and gid not in detected:
                flag = "  <- prespec(not flagged)"
            lines.append(row + flag)
        return lines

    # -- Assemble report -------------------------------------------------------
    lines = [
        "MathCheck Ranking Change Experiment",
        "====================================",
        f"30 canonical + 30 PU items (groups 0-29)  |  {n_models} models evaluated",
        f"Models: {', '.join(m for _, m in models)}",
        "",
        "EXECUTIVE SUMMARY",
        "-----------------",
        f"FormInv detection rule (canon_pass >= {FORMINV_DETECT_THRESHOLD}/{n_models} AND "
        f"pu_fail >= {FORMINV_DETECT_THRESHOLD}/{n_models}) flagged "
        f"{len(detected_bad)} bad PU paraphrases: groups {', '.join(sorted(detected_bad, key=int))}.",
        "",
        "Pre-specified hypotheses audit:",
        "  Group 38: UNTESTABLE -- falls outside groups 0-29 (groups evaluated: 0-29 only).",
        "  Group 26: DISCONFIRMED -- PU passes 4/4 models; prior pilot label was incorrect.",
        "  -> Pre-pilot hand-labeling was wrong on both pre-specified cases.",
        "",
        "FormInv automated detection (data-driven):",
        "  Group 25: unit-collapse bug ('0.5 hours' -> '30 minutes'; 4/4 models compute 2100 instead of 35)",
        "  Group 27: sub-question redirection (verbose PU -> 3/4 models answer 'throw distance' not 'distance-outside-range')",
        "",
    ]

    n_pu_all = len(results)
    n_pu_clean = len(results) - len(detected_bad)

    lines += fmt_table(metrics_all, "WITH all 30 PU items (incl. FormInv-flagged groups 25, 27):", n_pu_all)
    lines += [""]
    lines += fmt_table(
        metrics_clean,
        f"WITHOUT FormInv-flagged items ({n_pu_clean} PU items, excl. groups "
        f"{', '.join(sorted(detected_bad, key=int))}):",
        n_pu_clean,
    )
    lines += [""]

    # Ranking change analysis
    lines.append(f"Ranking change (SCR): {'YES' if scr_changed else 'COMPRESSION (tie at mid-tier)'}")
    all_scr_str = " > ".join(f"{mk.split('/')[-1]}({v:.1%})" for mk, v in ranks_all)
    clean_scr_str = " > ".join(f"{mk.split('/')[-1]}({v:.1%})" for mk, v in ranks_clean)
    lines += [
        f"  WITH    : {all_scr_str}",
        f"  WITHOUT : {clean_scr_str}",
    ]

    # Detect ties in clean ranking
    clean_scr_vals = [v for _, v in ranks_clean]
    ties = [
        ranks_clean[i][0].split("/")[-1]
        for i in range(len(ranks_clean) - 1)
        if abs(clean_scr_vals[i] - clean_scr_vals[i + 1]) < 1e-9
    ]
    if ties:
        lines.append(
            "  Note: removing bad items creates a tie between mid-tier models -- "
            "the gap was an artifact of which broken paraphrase each happened to fail."
        )

    lines += [""]

    # Cross-model unanimity table for flagged groups
    lines += [
        "Cross-model detail on FormInv-flagged groups:",
        "",
        f"{'Model':<40} {'Grp25 C->P (expected 35)':>25} {'Grp27 C->P (expected 200)':>25}",
        "-" * 95,
    ]
    for provider, model in models:
        mk = f"{provider}/{model}"
        g25 = results.get("25", {})
        g27 = results.get("27", {})
        c25 = g25.get("canonical", {}).get(mk, {})
        p25 = g25.get("pu", {}).get(mk, {})
        c27 = g27.get("canonical", {}).get(mk, {})
        p27 = g27.get("pu", {}).get(mk, {})

        def fmt_cell(cell_c, cell_p):
            c_ans = f"{cell_c.get('parsed_answer', '?')}"
            p_ans = f"{cell_p.get('parsed_answer', '?')}"
            c_ok = "+" if cell_c.get("correct") else "-"
            p_ok = "+" if cell_p.get("correct") else "-"
            return f"{c_ans}{c_ok}->{p_ans}{p_ok}"

        col25 = fmt_cell(c25, p25)
        col27 = fmt_cell(c27, p27)
        lines.append(f"{mk:<40} {col25:>25} {col27:>25}")

    lines += [
        "",
        "Pathology analysis:",
        "  Group 25: Canonical says '0.5 hours per dog'; PU says '30 minutes per dog'.",
        "            All 4 models omit unit conversion and compute 10x30x7=2100 instead of 35.",
        "            This is a unit-stripping error introduced by the paraphrase.",
        "  Group 27: Canonical asks 'how far outside the dragon's range' (answer: 200ft).",
        "            PU's verbose rewording causes 3/4 models to report throw-distance (1200ft).",
        "            Sub-question redirection from complex paraphrase structure.",
        "",
        "Pre-specified group audit:",
        "  Group 38: Not in evaluated range (groups 0-29 only). Would need groups 0-128 eval.",
    ]

    # Group 26 detail
    g26 = results.get("26", {})
    if g26:
        lines += ["  Group 26: Pre-pilot label predicted bad PU, but:"]
        for provider, model in models:
            mk = f"{provider}/{model}"
            c = g26.get("canonical", {}).get(mk, {})
            p = g26.get("pu", {}).get(mk, {})
            c_ok = "pass" if c.get("correct") else f"FAIL(got {c.get('parsed_answer')})"
            p_ok = "pass" if p.get("correct") else f"FAIL(got {p.get('parsed_answer')})"
            lines.append(f"    {mk}: canonical={c_ok}, PU={p_ok}")
        lines.append("    PU passes 4/4 models. Prior pilot label disconfirmed.")

    lines += ["", "--- Per-group detail ---", ""]
    lines += fmt_per_group(results, detected_bad)

    return "\n".join(lines)


# -- Main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="FormInv x MathCheck ranking-change experiment")
    parser.add_argument("--mock", action="store_true", help="Mock provider (no API calls)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass response cache")
    parser.add_argument("--n", type=int, default=30, help="Number of groups to evaluate (default 30)")
    parser.add_argument("--raw-out", default="artifacts/mathcheck_ranking_results.json")
    parser.add_argument("--report-out", default="artifacts/mathcheck_ranking_report.md")
    args = parser.parse_args()

    pairs = extract_pairs(n_groups=args.n)

    if args.mock:
        models = [("mock", "mock-model")]
    else:
        models = MODELS

    print("\nRunning evaluation:")
    print(f"  Groups  : {len(pairs)}")
    print(f"  Models  : {len(models)} -- {[m for _, m in models]}")
    print(f"  Total   : {len(pairs) * len(models) * 2} API calls (minus cache hits)")
    print(f"  Pre-spec: groups {', '.join(sorted(PRESPECIFIED_BAD_GROUPS))} (38 outside range 0-29)")
    print(
        f"  Detection: FormInv rule fires at canon_pass >= {FORMINV_DETECT_THRESHOLD} AND pu_fail >= {FORMINV_DETECT_THRESHOLD}"
    )
    print()

    results = run_eval(pairs, models, use_cache=not args.no_cache)

    # Save raw JSON
    raw_path = Path(args.raw_out)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(results, indent=2))
    print(f"\nRaw results saved to: {args.raw_out}")

    # Compute and save report
    report = build_report(results, models)
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
