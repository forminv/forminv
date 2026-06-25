# Reproducing the FormInv results

Every table and figure in the paper is regenerated from the cached model
responses shipped in `artifacts/` and `data/raw/` -- so the core results reproduce
**offline, without API keys**. Re-running an evaluation from scratch (to refresh
or extend the cache) requires the provider keys listed below.

## Environment

```bash
# install uv: https://astral.sh/uv
uv sync                 # runtime deps + forminv (editable)
uv sync --extra eval    # + provider SDKs (only needed to call live models)
```

API keys (only for live re-evaluation; each provider uses its **own** key):

| Provider  | Env var            |
|-----------|--------------------|
| OpenAI    | `OPENAI_API_KEY`   |
| Anthropic | `ANTHROPIC_API_KEY`|
| DeepSeek  | `DEEPSEEK_API_KEY` |
| Gemini    | `GOOGLE_API_KEY`   |

## Result -> command map

All commands are run from the repo root. Outputs land in `artifacts/`.

| Paper element | Command | Artifact |
|---|---|---|
| Sec. 5 Base evaluation (9 models) | `uv run --extra eval python scripts/run_all_models.py --dataset data/generated/forminv_v3_50.jsonl` | `artifacts/all_models_results.json` |
| Sec. 6.2 Mechanical-floor decomposition (permutation null) | `uv run python scripts/scr_decomposition.py` | `artifacts/scr_decomposition.json` |
| Sec. 6.3 Conditional Inconsistency Rate (audit-clean, all-correct subset) | `uv run python scripts/cir_rigorous.py` | `artifacts/cir_rigorous.json` |
| Sec. 6.3 CIR replication on the 103-theorem edition (from cache, no API) | `uv run python scripts/cir_rigorous.py --data data/generated/forminv_v3_103.jsonl --flagged none --out artifacts/cir_rigorous_103.json` | `artifacts/cir_rigorous_103.json` |
| Sec. 6.3 Multi-sample CIR -- full panel (n=8, T=0.7) | `uv run --extra eval python scripts/multisample_cir_multimodel.py --n 8 --workers 8` | `artifacts/multisample_cir_multimodel.json` |
| Sec. 6.3 Multi-sample CIR -- Claude only (n=8, T=0.7) | `uv run --extra eval python scripts/multisample_cir.py` | `artifacts/multisample_cir.json` |
| Sec. 5 Balanced TRUE/FALSE controls (subtle siblings) | `uv run --extra eval python scripts/balanced_eval.py --sib data/generated/false_siblings_subtle_50.json` | `artifacts/balanced_eval_subtle_full.json` |
| Paraphrase quality audit (leave-one-out) | `uv run python scripts/leave_one_out_audit.py` | `artifacts/leave_one_out_audit.json` |
| Figures | `uv run python scripts/generate_figures.py` | `paper/figures/` |

## Notes on reproducibility

- **Caching.** Live evaluations cache every response under `data/raw/` keyed by
  model, prompt, and (for multi-sample) sample index. Re-running reuses the cache;
  pass `--no-cache` to force fresh calls. Multi-sample uses temperature 0.7, so
  fresh samples will differ at the individual-draw level but reproduce the
  aggregate CIR within the reported confidence intervals.
- **Reasoning models** (`o4-mini`, `deepseek-reasoner`) ignore temperature, so the
  multi-model multi-sample script runs them at `n=1` automatically.
- **Seeds.** Bootstrap confidence intervals use a fixed RNG seed; dataset items
  carry a `generation_seed` (42) for build reproducibility.
- **Determinism.** The permutation null and bootstrap are seeded and fully
  deterministic given the cached responses.
