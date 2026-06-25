#!/usr/bin/env python3
"""
Conditional Inconsistency Rate (CIR) -- the difficulty-controlled invariance metric
that makes the "SCR is just p^k" critique unstatable.

Raw SCR conflates difficulty (hard theorem -> low p -> low SCR, mechanical) with
inconsistency (model flips a KNOWN-correct answer under rewording). CIR conditions
on knowledge:

  For each theorem the model answers correctly in CANONICAL form, does it flip to a
  WRONG answer on >=1 semantically-equivalent paraphrase?

  CIR = #(canonical-correct theorems with >=1 paraphrase flip) / #(canonical-correct theorems)
  PFR = #(wrong paraphrases of canonical-correct theorems) / #(paraphrases of canonical-correct theorems)

Because we condition on the model being correct on the canonical statement, CIR is
NOT a function of marginal accuracy: two models at equal accuracy can have CIR 0 or
high depending on WHERE their errors fall. We prove this empirically via the
accuracy-vs-CIR rank correlation and explicit equal-accuracy / different-CIR pairs,
and test each CIR against a within-model permutation null.
"""

import hashlib
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data/raw/response_cache")
DATA = os.path.join(ROOT, "data/generated/forminv_v3_50.jsonl")
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


def load():
    canon, paras = {}, defaultdict(list)  # canon[tid]=item ; paras[tid]=[items]
    for line in open(DATA):
        d = json.loads(line)
        if "canonical" in d["id"]:
            canon[d["theorem_id"]] = d
        else:
            paras[d["theorem_id"]].append(d)
    return canon, paras


def correctness(model, d):
    prompt = USER_TEMPLATE.format(question=d["nl_question"])
    f = os.path.join(CACHE, ckey(model, prompt) + ".json")
    if not os.path.exists(f):
        return None
    resp = json.load(open(f))
    parsed = resp.get("parsed") or parse_label(resp.get("raw", ""))
    if parsed is None:
        return None
    return int(parsed == d["ground_truth"])


def analyze(model, canon, paras, n_perm=20000, seed=0):
    # per-theorem: canonical correctness + paraphrase correctness vector
    rows = []
    all_para_flat = []
    for tid, cd in canon.items():
        cc = correctness(model, cd)
        pv = [correctness(model, p) for p in paras.get(tid, [])]
        pv = [x for x in pv if x is not None]
        if cc is None or len(pv) < 2:
            continue
        rows.append((tid, cc, pv))
        all_para_flat += pv
    acc_items = [cc for _, cc, _ in rows] + all_para_flat
    accuracy = float(np.mean(acc_items))
    # SCR (all paraphrases correct), unconditional
    scr = float(np.mean([int(all(pv)) for _, _, pv in rows]))
    # conditional on canonical-correct
    cond = [(tid, pv) for tid, cc, pv in rows if cc == 1]
    if not cond:
        return None
    flips = [int(not all(pv)) for _, pv in cond]
    cir = float(np.mean(flips))
    cond_paras = [x for _, pv in cond for x in pv]
    pfr = float(1 - np.mean(cond_paras))
    # permutation null for CIR: shuffle paraphrase-correct labels across the
    # canonical-correct pool, preserving group sizes and overall paraphrase acc.
    rng = np.random.default_rng(seed)
    sizes = [len(pv) for _, pv in cond]
    labels = np.array(cond_paras)
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(labels)
        idx = 0
        s = 0
        for sz in sizes:
            if not labels[idx : idx + sz].all():
                s += 1
            idx += sz
        null[i] = s / len(sizes)
    p_val = float((null <= cir).mean())  # observed flips >= null? we test EXCESS clustering -> obs CIR vs null
    return dict(
        model=model,
        accuracy=accuracy,
        scr=scr,
        cir=cir,
        pfr=pfr,
        n_cond=len(cond),
        cir_null=float(null.mean()),
        cir_excess=float(cir - null.mean()),
        p_value=p_val,
    )


def main():
    canon, paras = load()
    rows = [r for r in (analyze(m, canon, paras) for m in MODELS) if r]
    rows.sort(key=lambda r: -r["accuracy"])
    hdr = f"{'model':22} {'acc':>6} {'SCR':>6} {'CIR':>6} {'PFR':>6} {'CIRnull':>7} {'excess':>7} {'nCond':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['model']:22} {r['accuracy'] * 100:5.1f}% {r['scr'] * 100:5.1f}% "
            f"{r['cir'] * 100:5.1f}% {r['pfr'] * 100:5.1f}% {r['cir_null'] * 100:6.1f}% "
            f"{r['cir_excess'] * 100:+6.1f}% {r['n_cond']:5d}"
        )
    acc = np.array([r["accuracy"] for r in rows])
    scr = np.array([r["scr"] for r in rows])
    cir = np.array([r["cir"] for r in rows])
    print("\n--- Orthogonality to accuracy (the anti-mechanical proof) ---")
    rho_scr = spearmanr(acc, scr).statistic
    rho_cir = spearmanr(acc, cir).statistic
    print(f"Spearman rho(accuracy, SCR) = {rho_scr:+.2f}")
    print(f"Spearman rho(accuracy, CIR) = {rho_cir:+.2f}   (near 0 => invariance is a distinct axis)")
    # equal-accuracy / different-invariance pairs
    print("\nEqual-ish accuracy, divergent invariance (|dAcc|<=1.5pp):")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            da = abs(rows[i]["accuracy"] - rows[j]["accuracy"]) * 100
            dc = abs(rows[i]["cir"] - rows[j]["cir"]) * 100
            if da <= 1.5 and dc >= 8:
                print(f"  {rows[i]['model']} vs {rows[j]['model']}: dAcc={da:.1f}pp  dCIR={dc:.1f}pp")
    json.dump(rows, open(os.path.join(ROOT, "artifacts/conditional_inconsistency.json"), "w"), indent=2)
    print("\nWrote artifacts/conditional_inconsistency.json")


if __name__ == "__main__":
    main()
