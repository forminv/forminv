#!/usr/bin/env python3
"""
Leave-GPT-4o-out robustness for the cross-model unanimity audit (concern #3).

The audit flags a paraphrase as SUSPECT when >= THRESH of the panel get it wrong
(the signal: when many independent models all fail an item, the item is more likely
broken than every model being wrong). Reviewer concern: GPT-4o both *generates* the
paraphrases and *votes* on them. We test whether the flagged set is robust to
removing GPT-4o from the voting panel -- reconstructing every vote from the cache.
"""

import hashlib
import json
import os
from collections import defaultdict

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


def main():
    items = [json.loads(l) for l in open(DATA)]
    items = [d for d in items if "canonical" not in d["id"]]
    # votes[item_id][model] = correct(1/0)
    votes = defaultdict(dict)
    for d in items:
        prompt = USER_TEMPLATE.format(question=d["nl_question"])
        for m in MODELS:
            f = os.path.join(CACHE, ckey(m, prompt) + ".json")
            if not os.path.exists(f):
                continue
            resp = json.load(open(f))
            parsed = resp.get("parsed") or parse_label(resp.get("raw", ""))
            if parsed is None:
                continue
            votes[d["id"]][m] = int(parsed == d["ground_truth"])

    def flagged(panel, thresh_frac=0.6):
        """items where >= thresh_frac of panel-with-a-vote got it WRONG."""
        out = set()
        for iid, vd in votes.items():
            pv = [vd[m] for m in panel if m in vd]
            if len(pv) < 3:
                continue
            wrong_frac = 1 - sum(pv) / len(pv)
            if wrong_frac >= thresh_frac:
                out.add(iid)
        return out

    full = flagged(MODELS)
    no_gpt4o = flagged([m for m in MODELS if m != "gpt-4o"])
    inter = full & no_gpt4o
    print(f"Items with votes        : {len(votes)}")
    print(f"Flagged (full 8-panel)  : {len(full)}")
    print(f"Flagged (no GPT-4o)     : {len(no_gpt4o)}")
    if full:
        print(f"Retained without GPT-4o : {len(inter)}/{len(full)} = {100 * len(inter) / len(full):.0f}%")
    # also: does GPT-4o's own vote ever flip a flag? count items flagged only because of gpt-4o
    only_with = full - no_gpt4o
    only_without = no_gpt4o - full
    print(f"Flagged ONLY with GPT-4o: {len(only_with)}  | ONLY without GPT-4o: {len(only_without)}")
    json.dump(
        {
            "n_items": len(votes),
            "flagged_full": sorted(full),
            "flagged_no_gpt4o": sorted(no_gpt4o),
            "retention": (len(inter) / len(full) if full else None),
        },
        open(os.path.join(ROOT, "artifacts/leave_one_out_audit.json"), "w"),
        indent=2,
    )
    print("\nWrote artifacts/leave_one_out_audit.json")


if __name__ == "__main__":
    main()
