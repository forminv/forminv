---
license: cc-by-4.0
language:
- en
pretty_name: FormInv
size_categories:
- n<1K
task_categories:
- question-answering
- text-classification
task_ids:
- multiple-choice-qa
- natural-language-inference
annotations_creators:
- expert-generated
- machine-generated
language_creators:
- machine-generated
- expert-generated
source_datasets:
- extended|mathlib
tags:
- mathematical-reasoning
- semantic-invariance
- theorem-proving
- robustness
- evaluation
- paraphrase
- lean4
configs:
- config_name: all
  default: true
  data_files:
  - split: test
    path: data/all/test-*.parquet
- config_name: canonical
  data_files:
  - split: test
    path: data/canonical/test-*.parquet
- config_name: surface
  data_files:
  - split: test
    path: data/surface/test-*.parquet
dataset_info:
- config_name: all
  features:
  - name: id
    dtype: string
  - name: theorem_id
    dtype: string
  - name: lean4_statement
    dtype: string
  - name: mathlib_name
    dtype: string
  - name: domain
    dtype: string
  - name: tier
    dtype: string
  - name: family
    dtype: string
  - name: nl_question
    dtype: string
  - name: canonical_nl
    dtype: string
  - name: ground_truth
    dtype: string
  - name: verification_method
    dtype: string
  - name: generation_seed
    dtype: int64
  splits:
  - name: test
    num_bytes: <FILL after parquet>
    num_examples: 760
  download_size: <FILL after parquet>
  dataset_size: <FILL after parquet>
- config_name: canonical
  features:
  - name: id
    dtype: string
  - name: theorem_id
    dtype: string
  - name: lean4_statement
    dtype: string
  - name: mathlib_name
    dtype: string
  - name: domain
    dtype: string
  - name: tier
    dtype: string
  - name: family
    dtype: string
  - name: nl_question
    dtype: string
  - name: canonical_nl
    dtype: string
  - name: ground_truth
    dtype: string
  - name: verification_method
    dtype: string
  - name: generation_seed
    dtype: int64
  splits:
  - name: test
    num_bytes: <FILL after parquet>
    num_examples: 103
  download_size: <FILL after parquet>
  dataset_size: <FILL after parquet>
- config_name: surface
  features:
  - name: id
    dtype: string
  - name: theorem_id
    dtype: string
  - name: lean4_statement
    dtype: string
  - name: mathlib_name
    dtype: string
  - name: domain
    dtype: string
  - name: tier
    dtype: string
  - name: family
    dtype: string
  - name: nl_question
    dtype: string
  - name: canonical_nl
    dtype: string
  - name: ground_truth
    dtype: string
  - name: verification_method
    dtype: string
  - name: generation_seed
    dtype: int64
  splits:
  - name: test
    num_bytes: <FILL after parquet>
    num_examples: 657
  download_size: <FILL after parquet>
  dataset_size: <FILL after parquet>
---

# Dataset Card for FormInv

## Table of Contents
- [Dataset Description](#dataset-description)
- [Dataset Structure](#dataset-structure)
- [Dataset Creation](#dataset-creation)
- [Considerations for Using the Data](#considerations-for-using-the-data)
- [Additional Information](#additional-information)

## Dataset Description

- **Repository:** https://github.com/forminv/forminv
- **Paper:** FormInv: A Measurement Protocol for Semantic Invariance in Mathematical
  Reasoning Benchmarks (arXiv:2605.29001)
- **Point of Contact:** Noel Thomas <noel.thomas@mbzuai.ac.ae>

### Dataset Summary

FormInv measures **semantic invariance**: whether an LLM's verdict on a mathematical
claim stays constant under meaning-preserving rewording. The primary edition (`v3_103`)
contains **103 theorems** drawn from Lean 4 / Mathlib, each rendered as a *canonical*
natural-language statement plus several *surface* paraphrase families (e.g. syntactic
reordering, quantifier rephrasing, passive voice, notation substitution). Every item is a
TRUE/FALSE question carrying a ground-truth label that follows from the underlying Mathlib
theorem. The benchmark reports two headline metrics: the **Invariance Gap (IG)** -- the
accuracy spread across paraphrases of the same theorem -- and the **Self-Consistency Rate
(SCR)** -- the fraction of theorems on which a model returns an identical verdict to every
paraphrase.

Because all paraphrases preserve the meaning of a *true* Mathlib statement, every item in
the released editions is labelled `TRUE`. FormInv is therefore an **invariance probe**: a
non-invariant model is one whose verdict flips across reworded forms of the same true
claim. (FALSE controls are tracked in the project roadmap but are **not** part of these
editions; see [Other Known Limitations](#other-known-limitations).)

Two editions are published as Hub revisions of one repository:

| Edition | Theorems | Total rows (`all`) | Canonical | Surface |
|---|---|---|---|---|
| `v3_103` (primary) | 103 | **760** | 103 | 657 |
| `v3_50` | 50 | 366 | 50 | 316 |

### Supported Tasks and Leaderboards

- `text-classification` / `multiple-choice-qa`: answer TRUE or FALSE to each
  natural-language mathematical claim. Primary metrics: accuracy, **IG**, **SCR**.
- Leaderboard: a static results table is maintained in the GitHub README and at
  https://huggingface.co/spaces/forminv/forminv-leaderboard.

### Languages

English (`en`), with embedded Lean 4 / Mathlib formal statements.

## Dataset Structure

### Data Instances

A canonical item and one of its surface paraphrases (`v3_103`):

```json
{
  "id": "thm_0000_nat_prime_two_le_canonical",
  "theorem_id": "thm_0000_nat_prime_two_le",
  "lean4_statement": "theorem Nat.Prime.two_le : ∀ {p : ℕ}, p.Prime → 2 ≤ p",
  "mathlib_name": "Nat.Prime.two_le",
  "domain": "number_theory",
  "tier": "tier1",
  "family": "canonical",
  "nl_question": "Every prime number is at least 2 -- TRUE or FALSE?",
  "canonical_nl": "Every prime number is at least 2.",
  "ground_truth": "TRUE",
  "verification_method": "canonical",
  "generation_seed": 42
}
```

```json
{
  "id": "thm_0000_nat_prime_two_le_surface_syntactic",
  "theorem_id": "thm_0000_nat_prime_two_le",
  "lean4_statement": "theorem Nat.Prime.two_le : ∀ {p : ℕ}, p.Prime → 2 ≤ p",
  "mathlib_name": "Nat.Prime.two_le",
  "domain": "number_theory",
  "tier": "tier1",
  "family": "syntactic",
  "nl_question": "Is it true that a prime number is always at least 2?",
  "canonical_nl": "Every prime number is at least 2.",
  "ground_truth": "TRUE",
  "verification_method": "gpt4o_family_v2",
  "generation_seed": 42
}
```

### Data Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique item id (`<theorem_id>_<family>` / `<theorem_id>_surface_<family>`). |
| `theorem_id` | string | Groups all paraphrases of one theorem. |
| `lean4_statement` | string | Full Lean 4 theorem declaration (shared by all paraphrases of a theorem). |
| `mathlib_name` | string | Source Mathlib lemma name (provenance). |
| `domain` | string | Math domain: `number_theory`, `algebra`, `set_theory`, `combinatorics`, `analysis`, `order_theory`, `topology`. |
| `tier` | string | Difficulty tier: `tier1`-`tier3` (introductory → advanced). |
| `family` | string | Paraphrase family: `canonical`, `syntactic`, `quantifier`, `passive`, `notation`, `order`, `unpack`, `equivalent`, `variation`. |
| `nl_question` | string | The exact TRUE/FALSE question posed to the model. |
| `canonical_nl` | string | The canonical phrasing of the underlying claim (shared across a theorem's paraphrases). |
| `ground_truth` | string | `"TRUE"` or `"FALSE"`. All rows in the current editions are `"TRUE"` (true Mathlib statements). |
| `verification_method` | string | How the item was produced/verified: `canonical` (authored from the Mathlib statement) or `gpt4o_family_v2` (model-assisted paraphrase, screened). |
| `generation_seed` | int64 | Seed used to generate the item (reproducibility; `42` for these editions). |

> Note on `ground_truth`: stored as a plain string rather than a `ClassLabel`, because the
> current editions contain only the `TRUE` class. This keeps the schema fixed if FALSE
> controls are added in a future edition.

### Data Splits / Configs

Paraphrase strata are exposed as **configs** (the MMLU/GPQA pattern); each has a single
`test` split (FormInv is evaluation-only). Counts below are for the primary `v3_103`
edition (with `v3_50` in parentheses):

| Config | Split | # examples `v3_103` | # examples `v3_50` |
|---|---|---|---|
| `all` | test | 760 | 366 |
| `canonical` | test | 103 | 50 |
| `surface` | test | 657 | 316 |

The `surface` config is the union of all non-canonical families. Per-family row counts in
`v3_103`: `canonical` 103, `syntactic` 103, `quantifier` 103, `notation` 103, `unpack`
103, `equivalent` 103, `passive` 92, `variation` 26, `order` 24. (`passive`, `variation`,
and `order` apply only where the transformation is well-defined for the statement, hence
the lower counts.)

All items are evaluation-only (`test`); there is no training split. FormInv is a diagnostic
measurement protocol, not a fine-tuning corpus.

## Dataset Creation

### Curation Rationale

Standard math-reasoning benchmarks report a single accuracy per item, conflating
*capability* with *phrasing sensitivity*. FormInv isolates phrasing sensitivity by holding
the underlying claim fixed (a verified Lean 4 / Mathlib theorem) and varying only the
surface form, so that any change in a model's verdict is attributable to non-invariance.

### Source Data

#### Initial Data Collection and Normalization

Base theorems are sampled from Lean 4 / Mathlib; the source lemma name is recorded in
`mathlib_name` and the formal declaration in `lean4_statement` for provenance. A canonical
NL statement (`canonical_nl`) is authored to be faithful to the formal statement. Surface
paraphrases are then generated per family and screened to remain meaning-preserving.

#### Who are the source language producers?

Formal statements: the Mathlib community (Apache-2.0). NL canonical statements: the authors
(expert-generated, `verification_method == "canonical"`). Surface paraphrase families:
model-assisted generation with screening, recorded as `verification_method ==
"gpt4o_family_v2"`.

### Annotations

#### Annotation process

Ground-truth TRUE/FALSE labels follow directly from the formal Mathlib statement: every
released paraphrase is a meaning-preserving rewording of a *true* theorem and is therefore
labelled `TRUE`. Surface paraphrases are screened so that the transformation does not alter
truth value. A human baseline and inter-annotator agreement (Fleiss' kappa) are reported in the
paper.

#### Who are the annotators?

The authors and a small expert annotation pool (see paper, human-baseline study).

### Personal and Sensitive Information

None. The dataset contains only mathematical statements; no personal data.

## Considerations for Using the Data

### Social Impact of Dataset

FormInv is a diagnostic tool for the robustness of LLM mathematical reasoning. Positive
impact: it surfaces brittle, phrasing-dependent reasoning that single-accuracy benchmarks
hide. It is not intended for any high-stakes decision-making.

### Discussion of Biases

Coverage is limited to 103 theorems (50 in `v3_50`) across difficulty tiers `tier1`-`tier3`;
domains are weighted toward `algebra` and `number_theory`. Paraphrase families are a finite,
English-only set.

### Other Known Limitations

- **TRUE-only labels.** All released items are true Mathlib statements (`ground_truth ==
  "TRUE"`). The editions measure *consistency under rewording of true claims*; a model that
  answers TRUE unconditionally will appear perfectly invariant. Pair FormInv with a
  capability/accuracy benchmark, and interpret IG/SCR as invariance -- not correctness --
  signals. Matched FALSE controls are a roadmap item, not part of these editions.
- **Small item count** -- a deliberate, precision-focused design.
- **Model-assisted paraphrases.** Surface families (`verification_method ==
  "gpt4o_family_v2"`) may carry generator artifacts; mitigated by screening and tracked in
  `verification_method`. See `ERRATA.md` in the GitHub repo for corrections.

## Additional Information

### Dataset Curators

Nishal Thomas (Independent Researcher) and Noel Thomas (Mohamed bin Zayed
University of Artificial Intelligence, MBZUAI).

### Licensing Information

Released under **Creative Commons Attribution 4.0 International (CC-BY-4.0)**. Base formal
statements derive from **Lean 4 / Mathlib**, which is licensed under **Apache-2.0**; Mathlib
must be cited and its license respected when redistributing derivatives (the `mathlib_name`
and `lean4_statement` fields preserve this provenance). The accompanying evaluation code is
released separately under the **MIT License**.

### Citation Information

```bibtex
@misc{thomas2026forminv,
  title  = {FormInv: A Measurement Protocol for Semantic Invariance in
            Mathematical Reasoning Benchmarks},
  author = {Thomas, Nishal and Thomas, Noel},
  year   = {2026},
  eprint = {2605.29001},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```

Please also cite Mathlib (the source of the formal statements):

```bibtex
@inproceedings{mathlib2020,
  title     = {The {Lean} Mathematical Library},
  author    = {The mathlib Community},
  booktitle = {CPP},
  year      = {2020}
}
```

### Contributions

Issues and paraphrase-family contributions welcome via the GitHub repository.
