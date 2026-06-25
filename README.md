# FormInv

[![CI](https://github.com/forminv/forminv/actions/workflows/ci.yml/badge.svg)](https://github.com/forminv/forminv/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC--BY--4.0-green.svg)](LICENSE_DATA)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-forminv-yellow)](https://huggingface.co/datasets/forminv/forminv)

**A measurement protocol for *semantic invariance* in LLM mathematical reasoning.**

FormInv asks a question single-accuracy benchmarks cannot: when you reword a
mathematical claim without changing its meaning, does the model's verdict stay the
same? It pairs ~50-103 Lean 4 / Mathlib theorems with families of meaning-preserving
paraphrases and measures how often a model contradicts itself.

> **Paper:** *FormInv: A Measurement Protocol for Semantic Invariance in
> Mathematical Reasoning Benchmarks* ([arXiv:2605.29001](https://arxiv.org/abs/2605.29001)) --
> accepted poster, AI4Math @ ICML 2026.
> Nishal Thomas (Independent Researcher), Noel Thomas (MBZUAI).

## Overview

A model can be *accurate* yet *non-invariant* -- answering a theorem correctly in
one phrasing and incorrectly in an equivalent one. FormInv reports:

- **Conditional Inconsistency Rate (CIR)** -- the headline metric. Among theorems a
  model answers correctly *in canonical form*, the fraction where at least one
  meaning-preserving rewording flips the verdict. Because it conditions on items
  the model already gets right, CIR is **difficulty-controlled**: it cannot be
  explained away as a mechanical consequence of accuracy.
- **Semantic Consistency Rate (SCR)** -- the fraction of theorems on which every
  paraphrase gets the same (correct) answer. We show the raw cross-model SCR spread
  is *largely* a mechanical (p^k) consequence of accuracy, and isolate the part that
  is not (within-theorem failure clustering).
- **Invariance Gap (IG)** -- per-theorem accuracy spread across paraphrases.

### Headline result

On the theorems **every** model answers correctly in canonical form, a semantically
equivalent rewording still flips the verdict for a wide range of models -- and the
spread is invisible to accuracy. Multi-sample CIR (n=8, T=0.7), full panel:

| Model | CIR (systematic) | 95% CI |
|---|---:|---|
| DeepSeek V3 | 4.0% | [0.0, 10.0] |
| Gemini 2.5 Flash | 8.2% | [2.0, 16.3] |
| GPT-4o | 8.5% | [2.1, 17.0] |
| o4-mini | 10.0% | [2.0, 20.0] |
| GPT-4o-mini | 10.6% | [2.1, 21.3] |
| Claude Sonnet 4.6 | 10.6% | [2.1, 19.1] |
| DeepSeek R1 | 18.4% | [8.2, 30.6] |
| **Claude Haiku 4.5** | **35.4%** | **[22.9, 50.0]** |

Claude Haiku flips a *known-correct* answer ~9x as often as DeepSeek V3, a gap that
accuracy (87-97% on these items) cannot see. Multi-sampling shows these flips are
**systematic**, not temperature noise.

## Installation

Pick whichever environment manager you use. All three install the same pinned
dependency set (`uv.lock` / `requirements.txt`).

```bash
git clone https://github.com/forminv/forminv && cd forminv
```

uv (recommended; install from https://astral.sh/uv):

```bash
uv sync                 # runtime deps + forminv (editable)
uv sync --extra eval    # + provider SDKs, only needed to run live evaluations
uv sync --extra viz     # + matplotlib, only needed to regenerate figures
```

conda:

```bash
conda env create -f environment.yml
conda activate forminv
```

pip / venv:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .            # or: pip install -e ".[eval,viz]"
```

Run `make help` to see all targets (`make test`, `make figures`, `make repro`, ...).

## Dataset

The dataset is distributed on the Hugging Face Hub (Parquet, with a dataset viewer
and auto-generated Croissant metadata):

```python
from datasets import load_dataset

ds = load_dataset("forminv/forminv", "all")        # default config
ds = load_dataset("forminv/forminv", "canonical")  # canonical statements only
ds = load_dataset("forminv/forminv", "surface")    # surface paraphrases only
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique item id (`<theorem_id>_<family>`). |
| `theorem_id` | string | Groups all paraphrases of one theorem. |
| `lean4_statement` | string | Source Lean 4 / Mathlib declaration. |
| `mathlib_name` | string | Source Mathlib lemma name (provenance). |
| `domain`, `tier` | string | Math domain and difficulty tier. |
| `family` | string | Paraphrase family (`canonical`, `syntactic`, `quantifier`, ...). |
| `nl_question` | string | The exact TRUE/FALSE question posed. |
| `canonical_nl` | string | Canonical phrasing of the claim. |
| `ground_truth` | string | `TRUE` / `FALSE`. |
| `verification_method` | string | How the label/paraphrase was verified. |
| `generation_seed` | int | Build seed (reproducibility). |

Local JSONL editions live in `data/generated/` (`forminv_v3_50.jsonl`,
`forminv_v3_103.jsonl`). See [`hf_release/README.md`](hf_release/README.md) for the
full dataset card and [`docs/DATASHEET.md`](docs/DATASHEET.md) for the datasheet.

The underlying formal statements derive from **Lean 4 / Mathlib (Apache-2.0)**;
please cite Mathlib when using FormInv. Code is MIT; data is CC-BY-4.0.

## Quick start

```bash
uv run forminv eval --dataset data/generated/forminv_v3_50.jsonl --models all
uv run forminv cir --multisample --n 8 --workers 8     # multi-sample CIR, full panel
uv run forminv analyze --results artifacts/results.json
uv run forminv audit --results artifacts/results.json --threshold 6
uv run forminv selector --families unpack order --top-k 3
uv run forminv hf-export --input data/generated/forminv_v3_103.jsonl --out hf_build
```

Run `forminv --help` or `forminv <command> -h` for all options. Each provider uses
its own key: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`,
`GOOGLE_API_KEY`.

## Reproducing the paper

Every table and figure regenerates from the cached results shipped in `artifacts/`,
with no API access required:

```bash
make figures   # regenerate all figures into paper/
make repro     # recompute the headline numbers (CIR, SCR decomposition, audit)
```

See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the exact command behind each
table and figure (or run `forminv reproduce <target>`).

## Package layout

```
forminv/            Python package
  cli.py            `forminv` entry point (eval/build/analyze/audit/selector/cir/hf-export/reproduce)
  schemas.py        Core dataclasses (BaseTheorem, Paraphrase, ModelPrediction, ...)
  selector.py       Model recommendation by paraphrase family
  eval/             providers (multi-provider client), runner, analyze, audit
  generators/       theorem sources, paraphrase families, dataset builder
  metrics/          Invariance Gap, SCR, cross-model disagreement
scripts/            Experiment & reproduction scripts (see scripts/README.md)
data/               Dataset JSONL + splits
artifacts/          Cached eval outputs (the paper's evidence)
hf_release/         Hugging Face dataset card
docs/               Reproducing, datasheet, release notes
tests/              Offline unit/regression tests
paper/              LaTeX source
```

## Contributing

Contributions -- new paraphrase families, theorems, provider adapters, or errata --
are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`ERRATA.md`](ERRATA.md).

## Citation & license

```bibtex
@misc{thomas2026forminv,
  title         = {FormInv: A Measurement Protocol for Semantic Invariance in
                   Mathematical Reasoning Benchmarks},
  author        = {Thomas, Nishal and Thomas, Noel},
  year          = {2026},
  eprint        = {2605.29001},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

Code is released under the [MIT License](LICENSE); the dataset under
[CC-BY-4.0](LICENSE_DATA).
