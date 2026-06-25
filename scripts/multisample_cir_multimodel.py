#!/usr/bin/env python3
"""Multi-model, multi-sample Conditional Inconsistency Rate (CIR).

This extends ``multisample_cir.py`` from the Claude-only panel to the full
nine-model panel.  The metric and statistics are identical; only the API
dispatch is generalized to every provider.

Why multi-sample CIR
--------------------
A single temperature-0 evaluation cannot distinguish a *systematic* invariance
failure (the model reliably flips on a semantically equivalent rewording) from
a *stochastic* one (a borderline coin-flip).  We sample each item ``n`` times at
temperature ``T`` and, restricting to theorems the model answers correctly in
canonical form (``p_correct >= 0.75``), report the fraction with at least one
*systematic* paraphrase flip (a paraphrase with ``p_correct <= 0.25``).

Cost & caching
--------------
Every sample is cached by ``(model, prompt, sample_idx)`` in
``data/raw/multisample_cache`` using the *same* key scheme as the Claude-only
script, so previously collected Sonnet/Haiku samples are reused, not re-billed.

Reasoning models (``o4-mini``, ``deepseek-reasoner``) ignore temperature, so
multi-sampling is degenerate for them; they are automatically run at ``n=1``.

API key routing (each provider uses its own key)
------------------------------------------------
``openai`` -> ``OPENAI_API_KEY`` | ``anthropic`` -> ``ANTHROPIC_API_KEY`` |
``deepseek`` -> ``DEEPSEEK_API_KEY`` | ``gemini`` -> ``GOOGLE_API_KEY``
(OpenRouter is used *only* as a pre-authorized fallback for Gemini).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forminv.eval.providers import SYSTEM_PROMPT, USER_TEMPLATE, parse_label  # noqa: E402

MSCACHE = ROOT / "data/raw/multisample_cache"
DATA = ROOT / "data/generated/forminv_v3_50.jsonl"
LOO = ROOT / "artifacts/leave_one_out_audit.json"
MSCACHE.mkdir(parents=True, exist_ok=True)

# (provider, model, is_reasoning) -- reasoning models are forced to n=1.
PANEL = [
    ("openai", "gpt-4o", False),
    ("openai", "gpt-4o-mini", False),
    ("openai", "o4-mini", True),
    ("anthropic", "claude-sonnet-4-6", False),
    ("anthropic", "claude-haiku-4-5", False),
    ("deepseek", "deepseek-chat", False),
    ("deepseek", "deepseek-reasoner", True),
    ("gemini", "gemini-2.5-flash", False),
]


def cache_key(model: str, prompt: str, idx: int) -> str:
    """Return the 16-char cache id for one sample of ``(model, prompt)``."""
    return hashlib.md5(f"{model}::{prompt}::s{idx}".encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Per-provider single-sample calls.  Each returns a parsed "TRUE"/"FALSE"/None.
# --------------------------------------------------------------------------- #
def _call_openai(model: str, prompt: str, temp: float, reasoning: bool) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if reasoning:  # o-series ignore temperature and need a completion budget.
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["temperature"] = temp
        kwargs["max_tokens"] = 20
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _call_anthropic(model: str, prompt: str, temp: float, reasoning: bool) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=10,
        temperature=temp,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text if resp.content else ""


def _call_deepseek(model: str, prompt: str, temp: float, reasoning: bool) -> str:
    import openai

    client = openai.OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0 if reasoning else temp,
        max_tokens=4096 if reasoning else 20,
    )
    msg = resp.choices[0].message
    raw = msg.content or ""
    if not raw and getattr(msg, "reasoning_content", None):
        raw = msg.reasoning_content
    return raw


def _call_gemini(model: str, prompt: str, temp: float, reasoning: bool) -> str:
    """Native Google path first; OpenRouter is a pre-authorized Gemini-only fallback."""
    try:
        import google.genai as genai
        import google.genai.types as genai_types

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=temp,
                max_output_tokens=2048,
            ),
        )
        return resp.text or ""
    except Exception as exc:  # noqa: BLE001 -- fall back, but surface loudly.
        print(f"  [gemini native FAILED: {exc} -> OpenRouter fallback]")
        import openai

        client = openai.OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )
        resp = client.chat.completions.create(
            model=f"google/{model}",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
            max_tokens=20,
        )
        return resp.choices[0].message.content or ""


_DISPATCH = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "deepseek": _call_deepseek,
    "gemini": _call_gemini,
}


def fetch_sample(provider: str, model: str, reasoning: bool, prompt: str, idx: int, temp: float) -> str | None:
    """Return one parsed label for ``(prompt, idx)``, reading/writing the sample cache."""
    cf = MSCACHE / f"{cache_key(model, prompt, idx)}.json"
    if cf.exists():
        return json.loads(cf.read_text())["parsed"]
    raw = ""
    for attempt in range(4):
        try:
            raw = _DISPATCH[provider](model, prompt, temp, reasoning)
            break
        except Exception as exc:  # noqa: BLE001 -- transient API errors: backoff.
            if attempt == 3:
                print(f"  [WARN] {model} sample failed after retries: {exc}")
                raw = ""
            else:
                time.sleep(2 * (attempt + 1))
    parsed = parse_label(raw)
    cf.write_text(json.dumps({"parsed": parsed}))
    return parsed


def fetch_labels(
    provider: str, model: str, reasoning: bool, prompts: set[str], n: int, temp: float, workers: int
) -> dict[str, list[str | None]]:
    """Concurrently fetch ``n`` samples for each prompt. Returns ``{prompt: [labels]}``."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out: dict[str, list[str | None]] = {p: [None] * n for p in prompts}
    tasks = [(p, i) for p in prompts for i in range(n)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_sample, provider, model, reasoning, p, i, temp): (p, i) for (p, i) in tasks}
        done = 0
        for fut in as_completed(futs):
            p, i = futs[fut]
            out[p][i] = fut.result()
            done += 1
            if done % 400 == 0:
                print(f"  [{done}/{len(tasks)} samples]")
    return out


def p_correct(labels: list[str | None], gt: str) -> float | None:
    """Fraction of valid samples equal to the ground-truth label, or None."""
    valid = [l for l in labels if l is not None]
    if not valid:
        return None
    return sum(1 for l in valid if l == gt) / len(valid)


def compute_cir(per_thm: dict[str, tuple[float, list[float]]], seed: int = 1) -> dict:
    """Compute systematic/any CIR plus a bootstrap 95% CI over known theorems."""
    known = {t: pps for t, (cc, pps) in per_thm.items() if cc >= 0.75}
    has_sys = lambda pps: any(pc <= 0.25 for pc in pps)  # noqa: E731
    has_any = lambda pps: any(pc < 0.75 for pc in pps)  # noqa: E731
    if not known:
        return dict(n_known=0, cir_systematic=0.0, cir_any=0.0, ci_lo=0.0, ci_hi=0.0)
    arr = np.array([has_sys(pps) for pps in known.values()], dtype=float)
    rng = np.random.default_rng(seed)
    boot = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(10000)]
    return dict(
        n_known=len(known),
        cir_systematic=float(arr.mean()),
        cir_any=float(np.mean([has_any(pps) for pps in known.values()])),
        ci_lo=float(np.percentile(boot, 2.5) * 100),
        ci_hi=float(np.percentile(boot, 97.5) * 100),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="samples per item (temp models)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None, help="cap #theorems (pilot)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls per model")
    ap.add_argument("--models", default="all", help="comma-separated model names or 'all'")
    ap.add_argument("--out", default=str(ROOT / "artifacts/multisample_cir_multimodel.json"))
    args = ap.parse_args()

    panel = PANEL
    if args.models != "all":
        want = set(args.models.split(","))
        panel = [row for row in PANEL if row[1] in want]

    flagged = set(json.loads(LOO.read_text())["flagged_full"])
    canon: dict[str, dict] = {}
    paras: dict[str, list[dict]] = defaultdict(list)
    for line in DATA.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "canonical" in d["id"]:
            canon[d["theorem_id"]] = d
        else:
            paras[d["theorem_id"]].append(d)
    tids = list(canon.keys())[: args.limit] if args.limit else list(canon.keys())

    # Pre-build, per theorem, the canonical prompt and its audit-clean paraphrase
    # prompts (each paired with ground truth) so labels can be fetched in bulk.
    layout: dict[str, tuple[str, str, list[tuple[str, str]]]] = {}
    for tid in tids:
        cp = USER_TEMPLATE.format(question=canon[tid]["nl_question"])
        pp = [
            (USER_TEMPLATE.format(question=p["nl_question"]), p["ground_truth"])
            for p in paras.get(tid, [])
            if p["id"] not in flagged
        ]
        layout[tid] = (cp, canon[tid]["ground_truth"], pp)

    results: dict[str, dict] = {}
    for provider, model, reasoning in panel:
        n_eff = 1 if reasoning else args.n
        print(
            f"\nSampling {provider}/{model} (n={n_eff}, T={args.temp}, "
            f"reasoning={reasoning}) over {len(tids)} theorems ..."
        )
        all_prompts = {cp for cp, _, pp in layout.values()}
        all_prompts |= {q for _, _, pp in layout.values() for q, _ in pp}
        labels = fetch_labels(provider, model, reasoning, all_prompts, n_eff, args.temp, args.workers)
        per_thm: dict[str, tuple[float, list[float]]] = {}
        for tid, (cp, cgt, pp) in layout.items():
            cc = p_correct(labels[cp], cgt)
            pps = [pc for q, gt in pp if (pc := p_correct(labels[q], gt)) is not None]
            if cc is not None and len(pps) >= 2:
                per_thm[tid] = (cc, pps)
        stats = compute_cir(per_thm)
        stats["n_samples"] = n_eff
        stats["reasoning"] = reasoning
        results[f"{provider}/{model}"] = stats
        print(
            f"  {model}: known={stats['n_known']}  "
            f"CIR_sys={stats['cir_systematic'] * 100:.1f}% "
            f"[{stats['ci_lo']:.1f},{stats['ci_hi']:.1f}]  "
            f"CIR_any={stats['cir_any'] * 100:.1f}%"
        )

    print("\n" + "=" * 78)
    hdr = f"{'model':28} {'n':>2} {'nKnown':>6} {'CIRsys':>7} {'95% CI':>14} {'CIRany':>7}"
    print(hdr)
    print("-" * len(hdr))
    for mk, r in sorted(results.items(), key=lambda x: x[1]["cir_systematic"]):
        print(
            f"{mk:28} {r['n_samples']:>2} {r['n_known']:6d} "
            f"{r['cir_systematic'] * 100:6.1f}% "
            f"[{r['ci_lo']:4.1f},{r['ci_hi']:4.1f}] {r['cir_any'] * 100:6.1f}%"
        )

    Path(args.out).write_text(json.dumps({"n": args.n, "temp": args.temp, "results": results}, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
