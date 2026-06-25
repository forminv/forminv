# `scripts/` -- experiment & reproduction scripts

Standalone scripts (run with `uv run python scripts/<name>.py`). The package
itself lives in `forminv/`; these scripts orchestrate experiments around it.

For the exact command that regenerates each paper table/figure, see
[`../docs/REPRODUCING.md`](../docs/REPRODUCING.md).

## Canonical paper-result scripts

| Script | Purpose |
|---|---|
| `run_all_models.py` | Run the full 9-model panel over a dataset; base accuracy / IG / SCR. |
| `scr_decomposition.py` | Permutation null separating the mechanical (p^k) part of the SCR spread from genuine within-theorem clustering (Sec. 6.2). |
| `cir_rigorous.py` | Audit-clean Conditional Inconsistency Rate on the all-models-correct subset, with bootstrap CIs (Sec. 6.3). |
| `multisample_cir.py` | Multi-sample CIR (n=8, T=0.7), Claude panel -- systematic vs stochastic flips (Sec. 6.3). |
| `multisample_cir_multimodel.py` | Multi-sample CIR extended to the full provider panel (concurrent; reasoning models auto-run at n=1). |
| `conditional_inconsistency.py` | Loose (non-audit-cleaned) CIR -- provided for comparison. |
| `balanced_eval.py` | Balanced TRUE/FALSE study; `--sib` selects the subtle FALSE-sibling set (Sec. 5). |
| `leave_one_out_audit.py` | Leave-one-model-out paraphrase-quality audit (voter-circularity check). |
| `generate_figures.py` | Regenerate all paper figures from artifacts. |

## Dataset construction

| Script | Purpose |
|---|---|
| `build_dataset_v3.py` | Build the v3 paraphrase dataset (JSONL). |
| `build_hf_dataset.py` | Convert a FormInv JSONL into the Hugging Face Parquet/config layout. |
| `build_human_baseline_sample.py` | Sample items for the human-baseline annotation study. |
| `verify_false_siblings.py` | Independently re-verify numeric FALSE-sibling counterexamples. |

## Cross-benchmark & external audits

| Script | Purpose |
|---|---|
| `cross_benchmark_eval.py`, `run_cross_benchmark_folio.py`, `run_cross_benchmark_logiqa.py`, `write_cross_benchmark_summary.py` | Apply the invariance protocol to FOLIO / LogiQA. |
| `run_mathcheck_audit.py`, `run_mathcheck_ranking.py`, `run_mathcheck_ranking_129.py` | MathCheck cross-checks and ranking-change analysis. |
| `run_external_audit.py`, `run_mprobe.py` | External consistency audits. |
| `download_ntp_mathlib.py`, `extract_ntp_mathlib.py`, `convert_ntp_to_formInv.py`, `generate_ntp_paraphrases.py` | Next-theorem-proving / Mathlib ingestion pipeline. |

## Stats & utilities

| Script | Purpose |
|---|---|
| `bootstrap_ig_stability.py` | Bootstrap stability of IG estimates. |
| `compute_fleiss_kappa.py` | Inter-annotator agreement (Fleiss' kappa) for the human baseline. |
| `analyze_v3.py` | Summary analysis over a v3 results file. |
| `gate_check.py` | Metadata / quality gate validation. |
| `run_eval_v3.py`, `run_103_all_models.py`, `run_103_gemini_fix.py` | Edition-specific eval drivers. |
| `run_false_controls_all_models.py`, `run_false_controls_pilot.py` | FALSE-control evaluations across models. |
