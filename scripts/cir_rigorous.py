#!/usr/bin/env python3
"""
Rigorous Conditional Inconsistency Rate -- makes the matched-accuracy / inversion
claim airtight, all from cache (no API):

  (1) AUDIT-CLEAN: exclude unanimity-flagged paraphrases (a flip is the model's
      fault, not a broken test).
  (2) ALL-CORRECT SUBSET: restrict to theorems EVERY model answers correctly in
      canonical form, then compare flip rates on the SAME items (fully
      accuracy-controlled -- no difficulty confound possible).
  (3) Bootstrap 95% CIs over theorems for each model's CIR.
  (4) Paired bootstrap p-values for the two headline comparisons.
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data/raw/response_cache")
DATA = os.path.join(ROOT, "data/generated/forminv_v3_50.jsonl")
LOO = os.path.join(ROOT, "artifacts/leave_one_out_audit.json")
USER_TEMPLATE = "{question}\n\nAnswer TRUE or FALSE."
MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "deepseek-chat",
    "deepseek-reasoner",
    "gemini-2.5-flash",
]


def ckey(m, p):
    return hashlib.md5(f"{m}::{p}".encode()).hexdigest()[:16]


def parse_label(text):
    t = (text or "").strip().upper()
    for line in t.splitlines():
        line = line.strip()
        if line in ("TRUE", "FALSE"):
            return line
        if line.startswith("TRUE"):
            return "TRUE"
        if line.startswith("FALSE"):
            return "FALSE"
    if "TRUE" in t and "FALSE" not in t:
        return "TRUE"
    if "FALSE" in t and "TRUE" not in t:
        return "FALSE"
    return None


def correctness(model, d):
    f = os.path.join(CACHE, ckey(model, USER_TEMPLATE.format(question=d["nl_question"])) + ".json")
    if not os.path.exists(f):
        return None
    resp = json.load(open(f))
    parsed = resp.get("parsed") or parse_label(resp.get("raw", ""))
    return None if parsed is None else int(parsed == d["ground_truth"])


def main():
    ap = argparse.ArgumentParser(description="Rigorous CIR from cache (no API).")
    ap.add_argument("--data", default=DATA, help="dataset JSONL (default: v3_50)")
    ap.add_argument("--flagged", default=LOO, help="leave-one-out audit JSON, or 'none' to skip audit-cleaning")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts/cir_rigorous.json"))
    args = ap.parse_args()

    flagged = set() if args.flagged == "none" else set(json.load(open(args.flagged))["flagged_full"])
    canon, paras = {}, defaultdict(list)
    for line in open(args.data):
        d = json.loads(line)
        (canon.__setitem__(d["theorem_id"], d) if "canonical" in d["id"] else paras[d["theorem_id"]].append(d))

    # per model: theorem -> (canonical_correct, [paraphrase_correct on AUDIT-CLEAN])
    M = {}
    for model in MODELS:
        d_t = {}
        for tid, cd in canon.items():
            cc = correctness(model, cd)
            pv = [correctness(model, p) for p in paras.get(tid, []) if p["id"] not in flagged]
            pv = [x for x in pv if x is not None]
            if cc is not None and len(pv) >= 2:
                d_t[tid] = (cc, pv)
        M[model] = d_t

    # theorems where ALL models are canonical-correct AND have >=2 clean paraphrases
    common = set.intersection(*[set(d.keys()) for d in M.values()])
    all_correct = [t for t in common if all(M[m][t][0] == 1 for m in MODELS)]

    def cir_on(model, tids):
        flips = [int(not all(M[model][t][1])) for t in tids if M[model][t][0] == 1]
        return np.mean(flips) if flips else float("nan"), flips

    def boot_ci(model, tids, n=10000, seed=1):
        valid = [t for t in tids if M[model][t][0] == 1]
        arr = np.array([int(not all(M[model][t][1])) for t in valid])
        rng = np.random.default_rng(seed)
        bs = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
        return np.percentile(bs, 2.5) * 100, np.percentile(bs, 97.5) * 100

    print("=== AUDIT-CLEAN CIR (flagged paraphrases removed), bootstrap 95% CI ===")
    hdr = f"{'model':22} {'CIR':>6}  {'95% CI':>14}  {'nCanonCorrect':>13}"
    print(hdr)
    print("-" * len(hdr))
    cleanrows = {}
    for m in MODELS:
        cir, flips = cir_on(m, list(M[m].keys()))
        lo, hi = boot_ci(m, list(M[m].keys()))
        cleanrows[m] = cir
        print(f"{m:22} {cir * 100:5.1f}%  [{lo:4.1f}, {hi:4.1f}]  {len(flips):13d}")

    print(f"\n=== ALL-MODELS-CANONICAL-CORRECT subset: {len(all_correct)} theorems, SAME items ===")
    print("(fully accuracy-controlled: every model knows these on canonical)")
    hdr2 = f"{'model':22} {'flip rate on shared known theorems':>34}"
    print(hdr2)
    print("-" * len(hdr2))
    sub = {}
    for m in MODELS:
        fr, _ = cir_on(m, all_correct)
        sub[m] = fr
        print(f"{m:22} {fr * 100:33.1f}%")
    vals = [sub[m] for m in MODELS]
    print(
        f"\nSpread on identical known-correct theorems: "
        f"{100 * (max(vals) - min(vals)):.1f}pp  (min {100 * min(vals):.1f}%  max {100 * max(vals):.1f}%)"
    )

    # paired bootstrap for headline comparisons (on all_correct, paired by theorem)
    def paired_p(a, b, tids, n=20000, seed=2):
        da = np.array([int(not all(M[a][t][1])) for t in tids])
        db = np.array([int(not all(M[b][t][1])) for t in tids])
        diff = da - db
        rng = np.random.default_rng(seed)
        bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
        obs = diff.mean()
        # two-sided p that mean diff crosses 0
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        return obs * 100, min(p, 1.0)

    print("\n=== Headline comparisons (paired bootstrap on shared known theorems) ===")
    for a, b in [("claude-sonnet-4-6", "gpt-4o"), ("o4-mini", "gpt-4o")]:
        d, p = paired_p(a, b, all_correct)
        print(f"  {a} - {b}: DeltaflipRate = {d:+.1f}pp   p={p:.3f}")

    # Capability-vs-invariance correlations across models. The novelty test for CIR
    # is whether it merely relabels the accuracy ranking. We use CANONICAL accuracy
    # (pure capability, independent of paraphrase flipping) as the capability axis;
    # using overall accuracy would be circular (flips depress paraphrase accuracy).
    from scipy.stats import pearsonr, spearmanr

    def canonical_accuracy(model):
        vals = [correctness(model, cd) for cd in canon.values()]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    def model_scr(model):
        # SCR = fraction of (clean) theorems where canonical AND every paraphrase is correct.
        d_t = M[model]
        vals = [int(cc == 1 and all(pv)) for cc, pv in d_t.values()]
        return float(np.mean(vals)) if vals else float("nan")

    acc = {m: canonical_accuracy(m) for m in MODELS}
    scr = {m: model_scr(m) for m in MODELS}
    a = np.array([acc[m] for m in MODELS])
    cir_clean = np.array([cleanrows[m] for m in MODELS])
    cir_allc = np.array([sub[m] for m in MODELS])
    scr_a = np.array([scr[m] for m in MODELS])

    def corr(x, y):
        return dict(pearson=float(pearsonr(x, y)[0]), spearman=float(spearmanr(x, y)[0]))

    rho = {
        "acc_vs_clean_cir": corr(a, cir_clean),
        "acc_vs_allcorrect_cir": corr(a, cir_allc),
        "acc_vs_scr": corr(a, scr_a),
    }
    print(f"\n=== Capability(canonical acc)-vs-invariance correlations, {len(MODELS)} models ===")
    for k, v in rho.items():
        print(f"  {k:24} Pearson={v['pearson']:+.3f}  Spearman={v['spearman']:+.3f}")

    json.dump(
        {
            "clean_cir": {m: cleanrows[m] for m in MODELS},
            "n_all_correct": len(all_correct),
            "all_correct_flip": {m: sub[m] for m in MODELS},
            "canonical_accuracy": acc,
            "scr": scr,
            "correlations": rho,
        },
        open(args.out, "w"),
        indent=2,
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
