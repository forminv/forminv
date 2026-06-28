# FormInv

[![CI](https://github.com/forminv/forminv/actions/workflows/ci.yml/badge.svg)](https://github.com/forminv/forminv/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC--BY--4.0-green.svg)](LICENSE_DATA)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-forminv-yellow)](https://huggingface.co/datasets/forminv/forminv)

**Do LLMs give the same answer when you reword a math problem without changing its meaning?**

Usually not — and *which model looks best* can flip just from rephrasing. FormInv pairs
Lean 4 / Mathlib theorems with families of meaning-preserving paraphrases and measures how
often a model contradicts itself.

- **Paper:** accepted poster, AI4Math @ ICML 2026 — [arXiv:2605.29001](https://arxiv.org/abs/2605.29001)
- **Authors:** Nishal Thomas (Independent), Noel Thomas (MBZUAI)
- **Dataset:** [huggingface.co/datasets/forminv/forminv](https://huggingface.co/datasets/forminv/forminv)

## What it measures

- **CIR** — Conditional Inconsistency Rate (the headline). Of theorems a model gets right
  in canonical form, the fraction where a rewording flips its answer. Difficulty-controlled,
  so it can't be explained away by accuracy.
- **SCR** — Semantic Consistency Rate. Fraction of theorems where *every* paraphrase gets
  the same correct answer.
- **IG** — Invariance Gap. Per-theorem answer spread across paraphrases.

## Headline result

Even on theorems *every* model answers correctly, a reworded statement still flips the
verdict — invisibly to accuracy. Multi-sample CIR (n=8, T=0.7):

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

Claude Haiku flips a *known-correct* answer ~9× as often as DeepSeek V3 — a gap accuracy
(87–97% on these items) can't see, and that multi-sampling shows is systematic, not noise.

## Install

```bash
git clone https://github.com/forminv/forminv && cd forminv
uv sync                      # recommended
```

- **uv:** `uv sync` (add `--extra eval` for provider SDKs, `--extra viz` for figures)
- **pip:** `pip install -e ".[eval,viz]"`
- **conda:** `conda env create -f environment.yml`

## Quick start

```bash
uv run forminv eval --dataset data/generated/forminv_v3_50.jsonl --models all
uv run forminv cir --multisample --n 8                                # headline CIR
uv run forminv audit --results artifacts/results.json --threshold 6   # flag broken paraphrases
uv run forminv selector --families unpack order --top-k 3             # best model per family
```

Each provider uses its own key: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`,
`GOOGLE_API_KEY`. Run `forminv --help` for everything.

## Dataset

```python
from datasets import load_dataset
ds = load_dataset("forminv/forminv", "all")   # or "canonical" / "surface"
```

760 rows over 103 Lean 4 / Mathlib theorems × 8 paraphrase families. Local editions in
`data/generated/`. Full schema, datasheet, and field table: [`docs/DATASHEET.md`](docs/DATASHEET.md).

## Reproduce the paper

No API access needed — everything regenerates from cached results in `artifacts/`:

```bash
make figures   # all figures
make repro     # headline numbers (CIR, SCR decomposition, audit)
```

Per-table commands: [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

## Layout

```
forminv/      package — cli, eval, generators, metrics, selector
data/         dataset JSONL + splits
artifacts/    cached eval outputs (the paper's evidence)
paper/        LaTeX source
docs/         reproducing, datasheet
```

## Citation

```bibtex
@misc{thomas2026forminv,
  title  = {FormInv: A Measurement Protocol for Semantic Invariance in Mathematical Reasoning Benchmarks},
  author = {Thomas, Nishal and Thomas, Noel},
  year   = {2026},
  eprint = {2605.29001},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

Code MIT · data CC BY 4.0 · theorems from Lean 4 / Mathlib (Apache-2.0). Contributions and
errata welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).
