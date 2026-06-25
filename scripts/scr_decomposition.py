#!/usr/bin/env python3
"""
SCR mechanical-vs-structural decomposition.

Directly answers Reviewer Y7JT's strongest point: "SCR is a conjunction over
~7.4 paraphrases, so it is bounded below accuracy and shrinks like p^k by
construction." We test whether the observed SCR gap is MORE than what
independent per-item errors would produce.

Method
------
For each model we reconstruct per-(theorem, paraphrase) correctness from the
response cache (md5("{model}::{prompt}")[:16]). Then per theorem with k_t
non-canonical paraphrases:

  observed SCR  = fraction of theorems where ALL k_t paraphrases are correct
  mechanical    = E[SCR] under independence = mean_t prod_i p_i        (Poisson-binomial all-success)
                  with p_i = the model's per-item correctness probability
  null (perm)   = shuffle the model's correctness labels across all items,
                  recompute SCR; repeat -> null distribution of "SCR if errors
                  were not theorem-clustered".

If observed SCR >> mechanical/perm-null, failures are CORRELATED within a
theorem: SCR measures genuine per-theorem invariance, not p^k arithmetic.
We also report the "excess consistency" = observed - mechanical.
"""

import hashlib
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data/raw/response_cache")
DATA = os.path.join(ROOT, "data/generated/forminv_v3_50.jsonl")

USER_TEMPLATE = "{question}\n\nAnswer TRUE or FALSE."

MODELS = [
    ("gpt-4o", "openai"),
    ("gpt-4o-mini", "openai"),
    ("o4-mini", "openai"),
    ("claude-sonnet-4-6", "anthropic"),
    ("claude-haiku-4-5", "anthropic"),
    ("deepseek-chat", "deepseek"),
    ("deepseek-reasoner", "deepseek"),
    ("gemini-2.5-flash", "gemini"),
]


def ckey(model, prompt):
    return hashlib.md5(f"{model}::{prompt}".encode()).hexdigest()[:16]


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


def load_items():
    items = []
    with open(DATA) as f:
        for line in f:
            d = json.loads(line)
            if "canonical" in d["id"]:
                continue  # canonical excluded from SCR (matches eval pipeline)
            items.append(d)
    return items


def model_outcomes(model, items):
    """Return dict theorem_id -> list of 1/0 correctness (cached items only)."""
    by_thm = defaultdict(list)
    n_hit = 0
    for d in items:
        prompt = USER_TEMPLATE.format(question=d["nl_question"])
        f = os.path.join(CACHE, ckey(model, prompt) + ".json")
        if not os.path.exists(f):
            continue
        resp = json.load(open(f))
        parsed = resp.get("parsed") or parse_label(resp.get("raw", ""))
        if parsed is None:
            continue
        n_hit += 1
        gt = d["ground_truth"]  # "TRUE"/"FALSE"
        correct = int(parsed == gt)
        by_thm[d["theorem_id"]].append(correct)
    return by_thm, n_hit


def analyze(model, items, n_perm=20000, seed=0):
    by_thm, n_hit = model_outcomes(model, items)
    # keep theorems with >=2 paraphrases (SCR needs a conjunction)
    by_thm = {t: v for t, v in by_thm.items() if len(v) >= 2}
    if len(by_thm) < 5:
        return None
    flat = [c for v in by_thm.values() for c in v]
    p = np.mean(flat)  # per-item accuracy
    obs_scr = np.mean([int(all(v)) for v in by_thm.values()])
    # mechanical baseline: independence with per-item prob = p (global)
    mech = np.mean([p ** len(v) for v in by_thm.values()])
    # permutation null: break theorem clustering, keep group sizes + overall p
    rng = np.random.default_rng(seed)
    sizes = [len(v) for v in by_thm.values()]
    labels = np.array(flat)
    perm_scr = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(labels)
        idx = 0
        s = 0
        for sz in sizes:
            if labels[idx : idx + sz].all():
                s += 1
            idx += sz
        perm_scr[i] = s / len(sizes)
    p_val = float((perm_scr >= obs_scr).mean())
    return dict(
        model=model,
        n_thm=len(by_thm),
        n_items=n_hit,
        acc=float(p),
        obs_scr=float(obs_scr),
        mech_scr=float(mech),
        perm_mean=float(perm_scr.mean()),
        perm_p95=float(np.percentile(perm_scr, 95)),
        excess=float(obs_scr - mech),
        p_value=p_val,
    )


def main():
    items = load_items()
    print(f"Loaded {len(items)} non-canonical items, {len(set(d['theorem_id'] for d in items))} theorems\n")
    rows = []
    for model, _prov in MODELS:
        r = analyze(model, items)
        if r:
            rows.append(r)
    if not rows:
        print("No cached outcomes found.")
        return
    rows.sort(key=lambda r: -r["obs_scr"])
    hdr = f"{'model':22} {'acc':>6} {'obsSCR':>7} {'mechSCR':>8} {'excess':>7} {'permMean':>8} {'p':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['model']:22} {r['acc'] * 100:5.1f}% {r['obs_scr'] * 100:6.1f}% "
            f"{r['mech_scr'] * 100:7.1f}% {r['excess'] * 100:+6.1f}% "
            f"{r['perm_mean'] * 100:7.1f}% {r['p_value']:6.3f}"
        )
    print()
    # headline: does the SCR ranking gap survive after removing the mechanical floor?
    accs = np.array([r["acc"] for r in rows])
    obs = np.array([r["obs_scr"] for r in rows])
    mech = np.array([r["mech_scr"] for r in rows])
    print(
        f"SCR spread (max-min):            observed={100 * (obs.max() - obs.min()):.1f}pp  "
        f"mechanical={100 * (mech.max() - mech.min()):.1f}pp"
    )
    print(f"Accuracy spread (max-min):       {100 * (accs.max() - accs.min()):.1f}pp")
    print(f"Mean excess consistency:         {100 * np.mean(obs - mech):+.1f}pp (obs SCR above independence floor)")
    json.dump(rows, open(os.path.join(ROOT, "artifacts/scr_decomposition.json"), "w"), indent=2)
    print("\nWrote artifacts/scr_decomposition.json")


if __name__ == "__main__":
    main()
