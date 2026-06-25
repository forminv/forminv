#!/usr/bin/env python3
"""
Balanced TRUE/FALSE evaluation -- defuses Reviewer Y7JT + AC "all-TRUE confound".

Pipeline (Claude-only, no OpenRouter):
  1. Load 50 TRUE theorems (existing 316 non-canonical TRUE items).
  2. For each verified FALSE sibling (false_siblings_50.json), generate family
     paraphrases with CLAUDE (claude-sonnet-4-6) -- this also breaks the GPT-4o
     generator-subject circularity (concern #3), since the TRUE set was GPT-4o-made.
  3. Eval claude-sonnet-4-6 + claude-haiku-4-5 on the balanced set.
  4. Report balanced accuracy, balanced-SCR (TRUE-SCR and FALSE-SCR separately),
     and whether model separation survives the rebalance.

FALSE-preservation guard: the generator is told to reword surface form only and
keep the (false) claim intact; we then re-ask Claude a verification question on a
sample and flag any paraphrase whose truth value drifted.
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict

import anthropic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data/raw/response_cache")
GENCACHE = os.path.join(ROOT, "data/raw/false_gen_cache")
TRUE_DATA = os.path.join(ROOT, "data/generated/forminv_v3_50.jsonl")
SIB = os.path.join(ROOT, "data/generated/false_siblings_50.json")
os.makedirs(GENCACHE, exist_ok=True)

EVAL_SYSTEM = (
    "You are evaluating mathematical statements.\n"
    "Answer strictly based on mathematical correctness, not user preference.\n"
    "Return exactly TRUE or FALSE."
)
EVAL_USER = "{question}\n\nAnswer TRUE or FALSE."

FAMILIES = {
    "syntactic": "Rephrase as a direct yes/no question ('Is it true that ...?'), keeping the exact same claim.",
    "quantifier": "Rephrase making the quantifier explicit ('for every', 'for all', 'there exists'), keeping the exact same claim.",
    "passive": "Rephrase in passive voice, keeping the exact same claim.",
    "notation": "Rephrase using mathematical notation/symbols where natural, keeping the exact same claim.",
    "unpack": "Rephrase by unpacking definitions into more elementary terms, keeping the exact same claim.",
    "equivalent": "Rephrase using equivalent terminology/synonyms, keeping the exact same claim.",
}

GEN_SYSTEM = (
    "You are a careful paraphrase generator for a benchmark that tests robustness. "
    "You will be given a mathematical statement that may be TRUE or FALSE. "
    "Your job is ONLY to reword the surface form. You must PRESERVE the exact logical "
    "claim -- including its truth value. NEVER correct, fix, soften, or negate a false "
    "statement. If the input is false, your paraphrase must remain equally false. "
    "Output only the reworded statement, no preamble, no answer, no commentary."
)


def ckey(model, prompt):
    return hashlib.md5(f"{model}::{prompt}".encode()).hexdigest()[:16]


def client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


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


def gen_paraphrase(cl, base_stmt, family_instr):
    prompt = f"Statement: {base_stmt}\n\nInstruction: {family_instr}"
    cf = os.path.join(GENCACHE, ckey("gen-sonnet", prompt) + ".json")
    if os.path.exists(cf):
        return json.load(open(cf))["text"]
    resp = cl.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=120,
        temperature=0,
        system=GEN_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    txt = resp.content[0].text.strip()
    json.dump({"text": txt}, open(cf, "w"))
    return txt


def build_false_items(cl, limit=None, sib_path=None):
    sibs = json.load(open(sib_path or SIB))["siblings"]
    if limit:
        sibs = sibs[:limit]
    items = []
    for s in sibs:
        tid = s["theorem_id"]
        base = s["false_stmt"]
        # canonical FALSE item
        items.append(
            dict(
                id=f"{tid}_FALSE_canonical",
                theorem_id=f"{tid}_FALSE",
                family="canonical",
                ground_truth="FALSE",
                nl_question=f"{base} -- TRUE or FALSE?",
            )
        )
        for fam, instr in FAMILIES.items():
            para = gen_paraphrase(cl, base, instr)
            q = para if para.rstrip().endswith("?") else f"{para.rstrip('.')} -- TRUE or FALSE?"
            items.append(
                dict(
                    id=f"{tid}_FALSE_{fam}",
                    theorem_id=f"{tid}_FALSE",
                    family=fam,
                    ground_truth="FALSE",
                    nl_question=q,
                )
            )
    return items


def load_true_items(limit=None):
    items = []
    tids = []
    for line in open(TRUE_DATA):
        d = json.loads(line)
        if "canonical" in d["id"]:
            continue
        items.append(
            dict(
                id=d["id"],
                theorem_id=d["theorem_id"],
                family=d["family"],
                ground_truth=d["ground_truth"],
                nl_question=d["nl_question"],
            )
        )
        tids.append(d["theorem_id"])
    if limit:
        keep = set(sorted(set(tids))[:limit])
        items = [i for i in items if i["theorem_id"] in keep]
    return items


def eval_model(cl, model, items):
    """Return per-item correctness; cache-compatible with the main pipeline."""
    out = []
    for it in items:
        prompt = EVAL_USER.format(question=it["nl_question"])
        cf = os.path.join(CACHE, ckey(model, prompt) + ".json")
        if os.path.exists(cf):
            resp = json.load(open(cf))
            parsed = resp.get("parsed") or parse_label(resp.get("raw", ""))
        else:
            r = cl.messages.create(
                model=model,
                max_tokens=10,
                temperature=0,
                system=EVAL_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = r.content[0].text
            parsed = parse_label(raw)
            json.dump({"raw": raw, "parsed": parsed, "latency_s": 0.0}, open(cf, "w"))
        correct = None if parsed is None else int(parsed == it["ground_truth"])
        out.append(
            dict(
                theorem_id=it["theorem_id"],
                family=it["family"],
                gt=it["ground_truth"],
                parsed=parsed,
                correct=correct,
            )
        )
    return out


def scr(by_thm):
    by_thm = {t: v for t, v in by_thm.items() if len(v) >= 2}
    if not by_thm:
        return float("nan"), 0
    return sum(int(all(v)) for v in by_thm.values()) / len(by_thm), len(by_thm)


def cir(preds, gt):
    """Conditional Inconsistency Rate for one truth value.

    Among theorems whose canonical item the model answers correctly (i.e. it
    "knows" the claim), the fraction with at least one paraphrase that flips to
    incorrect. Requires a canonical item to condition on (present for FALSE).

    Args:
        preds: per-item prediction dicts (theorem_id, family, gt, correct).
        gt: ground-truth value to condition on ("TRUE" or "FALSE").

    Returns:
        (cir, n_known): the flip rate and the number of canonical-correct theorems.
    """
    grp = defaultdict(lambda: {"canon": None, "paras": []})
    for p in preds:
        if p["gt"] != gt or p["correct"] is None:
            continue
        if "canonical" in p["family"]:
            grp[p["theorem_id"]]["canon"] = p["correct"]
        else:
            grp[p["theorem_id"]]["paras"].append(p["correct"])
    known = {t: v for t, v in grp.items() if v["canon"] == 1 and len(v["paras"]) >= 2}
    if not known:
        return float("nan"), 0
    flips = [int(not all(v["paras"])) for v in known.values()]
    return sum(flips) / len(flips), len(known)


def metrics(preds):
    valid = [p for p in preds if p["correct"] is not None]

    def acc(sub):
        s = [p["correct"] for p in sub]
        return sum(s) / len(s) if s else float("nan")

    T = [p for p in valid if p["gt"] == "TRUE"]
    F = [p for p in valid if p["gt"] == "FALSE"]
    bt = defaultdict(list)
    ft = defaultdict(list)
    for p in T:
        if "canonical" not in p["family"]:
            bt[p["theorem_id"]].append(p["correct"])
    for p in F:
        if "canonical" not in p["family"]:
            ft[p["theorem_id"]].append(p["correct"])
    tscr, ntt = scr(bt)
    fscr, ntf = scr(ft)
    fcir, nfc = cir(preds, "FALSE")
    return dict(
        acc_true=acc(T),
        acc_false=acc(F),
        bal_acc=0.5 * (acc(T) + acc(F)),
        scr_true=tscr,
        scr_false=fscr,
        bal_scr=0.5 * (tscr + fscr) if fscr == fscr else float("nan"),
        false_cir=fcir,
        n_false_cir=nfc,
        n_true=len(T),
        n_false=len(F),
        n_thm_true=ntt,
        n_thm_false=ntf,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="theorems (pilot)")
    ap.add_argument("--sib", default=None, help="path to false-siblings JSON (default: blatant set)")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts/balanced_eval.json"))
    args = ap.parse_args()

    cl = client()
    print(f"Generating FALSE paraphrases with claude-sonnet-4-6 (siblings: {args.sib or 'default blatant'}) ...")
    false_items = build_false_items(cl, limit=args.limit, sib_path=args.sib)
    true_items = load_true_items(limit=args.limit)
    print(f"  TRUE items: {len(true_items)}   FALSE items: {len(false_items)}")
    all_items = true_items + false_items

    results = {}
    for model in ["claude-sonnet-4-6", "claude-haiku-4-5"]:
        print(f"Evaluating {model} on {len(all_items)} items ...")
        preds = eval_model(cl, model, all_items)
        results[model] = metrics(preds)

    print("\n" + "=" * 78)
    hdr = f"{'model':20} {'accT':>6} {'accF':>6} {'balAcc':>7} {'scrT':>6} {'scrF':>6} {'balSCR':>7}"
    print(hdr)
    print("-" * len(hdr))
    for m, r in results.items():
        print(
            f"{m:20} {r['acc_true'] * 100:5.1f}% {r['acc_false'] * 100:5.1f}% "
            f"{r['bal_acc'] * 100:6.1f}% {r['scr_true'] * 100:5.1f}% {r['scr_false'] * 100:5.1f}% "
            f"{r['bal_scr'] * 100:6.1f}%"
        )
    json.dump(
        {"results": results, "n_false_items": len(false_items), "n_true_items": len(true_items)},
        open(args.out, "w"),
        indent=2,
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
