#!/usr/bin/env python3
"""
Multi-sample Conditional Inconsistency Rate (the rigor upgrade).

Single-sample temp-0 CIR cannot tell a *systematic* invariance failure (the model
reliably flips on a rewording) from a *stochastic* one (a borderline coin-flip).
We sample each item n times at temperature T and:

  - per item: p_correct = fraction of samples correct; majority label.
  - a paraphrase is a SYSTEMATIC flip if p_correct <= 0.25 on a canonical-known
    theorem; a STOCHASTIC/borderline flip if 0.25 < p_correct < 0.75.
  - multi-sample CIR = among canonical-known theorems (canonical p_correct >= 0.75),
    fraction with >= 1 systematic paraphrase flip (audit-clean paraphrases only).
  - bootstrap 95% CI over theorems.

Claude-only (sonnet, haiku). Resumable: each sample cached by (model,prompt,idx).
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict

import anthropic
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSCACHE = os.path.join(ROOT, "data/raw/multisample_cache")
DATA = os.path.join(ROOT, "data/generated/forminv_v3_50.jsonl")
LOO = os.path.join(ROOT, "artifacts/leave_one_out_audit.json")
os.makedirs(MSCACHE, exist_ok=True)

SYSTEM = (
    "You are evaluating mathematical statements.\n"
    "Answer strictly based on mathematical correctness, not user preference.\n"
    "Return exactly TRUE or FALSE."
)
USER = "{question}\n\nAnswer TRUE or FALSE."
MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5"]


def ckey(model, prompt, idx):
    return hashlib.md5(f"{model}::{prompt}::s{idx}".encode()).hexdigest()[:16]


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


def sample_item(cl, model, question, n, temp):
    prompt = USER.format(question=question)
    labels = []
    for i in range(n):
        cf = os.path.join(MSCACHE, ckey(model, prompt, i) + ".json")
        if os.path.exists(cf):
            labels.append(json.load(open(cf))["parsed"])
            continue
        r = cl.messages.create(
            model=model,
            max_tokens=10,
            temperature=temp,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = parse_label(r.content[0].text)
        json.dump({"parsed": parsed}, open(cf, "w"))
        labels.append(parsed)
    return labels


def p_correct(labels, gt):
    valid = [l for l in labels if l is not None]
    if not valid:
        return None
    return sum(1 for l in valid if l == gt) / len(valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts/multisample_cir.json"))
    args = ap.parse_args()

    flagged = set(json.load(open(LOO))["flagged_full"])
    canon, paras = {}, defaultdict(list)
    for line in open(DATA):
        d = json.loads(line)
        (canon.__setitem__(d["theorem_id"], d) if "canonical" in d["id"] else paras[d["theorem_id"]].append(d))
    tids = list(canon.keys())
    if args.limit:
        tids = tids[: args.limit]

    cl = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = {}
    for model in MODELS:
        print(f"Sampling {model} (n={args.n}, T={args.temp}) ...")
        per_thm = {}
        for ti, tid in enumerate(tids):
            if ti % 10 == 0:
                print(f"  [{ti}/{len(tids)}]")
            cc = p_correct(
                sample_item(cl, model, canon[tid]["nl_question"], args.n, args.temp),
                canon[tid]["ground_truth"],
            )
            pps = []
            for p in paras.get(tid, []):
                if p["id"] in flagged:
                    continue
                pc = p_correct(sample_item(cl, model, p["nl_question"], args.n, args.temp), p["ground_truth"])
                if pc is not None:
                    pps.append(pc)
            if cc is not None and len(pps) >= 2:
                per_thm[tid] = (cc, pps)
        # metric
        known = {t: pps for t, (cc, pps) in per_thm.items() if cc >= 0.75}

        def has_sys_flip(pps):
            return any(pc <= 0.25 for pc in pps)

        def has_any_flip(pps):
            return any(pc < 0.75 for pc in pps)

        cir_sys = np.mean([has_sys_flip(pps) for pps in known.values()])
        cir_any = np.mean([has_any_flip(pps) for pps in known.values()])
        # bootstrap CI on systematic CIR
        keys = list(known.values())
        arr = np.array([has_sys_flip(pps) for pps in keys], dtype=float)
        rng = np.random.default_rng(1)
        bs = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(10000)]
        lo, hi = np.percentile(bs, 2.5) * 100, np.percentile(bs, 97.5) * 100
        results[model] = dict(
            n_known=len(known),
            cir_systematic=float(cir_sys),
            cir_any=float(cir_any),
            ci_lo=float(lo),
            ci_hi=float(hi),
        )
        print(
            f"  {model}: known={len(known)}  CIR_sys={cir_sys * 100:.1f}% "
            f"[{lo:.1f},{hi:.1f}]  CIR_any={cir_any * 100:.1f}%"
        )

    print("\n" + "=" * 70)
    hdr = f"{'model':22} {'nKnown':>6} {'CIRsys':>7} {'95% CI':>14} {'CIRany':>7}"
    print(hdr)
    print("-" * len(hdr))
    for m, r in results.items():
        print(
            f"{m:22} {r['n_known']:6d} {r['cir_systematic'] * 100:6.1f}% "
            f"[{r['ci_lo']:4.1f},{r['ci_hi']:4.1f}] {r['cir_any'] * 100:6.1f}%"
        )
    json.dump({"n": args.n, "temp": args.temp, "results": results}, open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
