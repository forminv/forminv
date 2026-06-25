# Datasheet for FormInv

This datasheet follows the *Datasheets for Datasets* questionnaire (Gebru et al., 2021) and
documents the FormInv benchmark as released on the Hugging Face Hub. It complements the
dataset card ([`../hf_release/README.md`](../hf_release/README.md)); where the two overlap,
the Hub card is authoritative.

**Editions covered:** `v3_103` (primary; 760 rows, 103 theorems) and `v3_50` (366 rows, 50
theorems). Row counts below refer to `v3_103` unless stated otherwise.

---

## 1. Motivation

**For what purpose was the dataset created?**
FormInv was created to measure **semantic invariance** in LLM mathematical reasoning:
whether a model's TRUE/FALSE verdict on a mathematical claim stays constant when the claim
is reworded in a meaning-preserving way. Standard math benchmarks report a single accuracy
per item and so conflate *capability* with *phrasing sensitivity*. FormInv isolates phrasing
sensitivity by holding the underlying claim fixed (a verified Lean 4 / Mathlib theorem) and
varying only the surface form, enabling two headline metrics: the **Invariance Gap (IG)**
and the **Self-Consistency Rate (SCR)**.

**Who created the dataset and on behalf of whom?**
Nishal Thomas (Independent Researcher) and Noel Thomas (Mohamed bin Zayed University of
Artificial Intelligence, MBZUAI; `noel.thomas@mbzuai.ac.ae`), as the research artifact
accompanying the FormInv paper (NeurIPS Datasets & Benchmarks submission).

**Who funded the creation of the dataset?**
See the Acknowledgements section of the paper.

---

## 2. Composition

**What do the instances represent?**
Each instance is a single natural-language TRUE/FALSE question about one mathematical claim,
together with provenance and metadata. Instances are grouped by `theorem_id`: one Mathlib
theorem yields one `canonical` rendering plus several `surface` paraphrases.

**How many instances are there in total?**
- `v3_103`: **760** rows over **103** theorems -- `all` 760, `canonical` 103, `surface` 657.
- `v3_50`: 366 rows over 50 theorems -- `all` 366, `canonical` 50, `surface` 316.

**Per-family counts (`v3_103`):** `canonical` 103, `syntactic` 103, `quantifier` 103,
`notation` 103, `unpack` 103, `equivalent` 103, `passive` 92, `variation` 26, `order` 24.
The three smaller families apply only where the transformation is well-defined for the
statement.

**Does the dataset contain all possible instances or is it a sample?**
It is a curated sample of Mathlib theorems (103 / 50), chosen for clarity of NL rendering
and coverage across domains and difficulty tiers. It is not an exhaustive or uniformly
random sample of Mathlib.

**What data does each instance consist of?**
Twelve fields (the exact released schema):

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique item id (`<theorem_id>_<family>`). |
| `theorem_id` | string | Groups all paraphrases of one theorem. |
| `lean4_statement` | string | Full Lean 4 theorem declaration. |
| `mathlib_name` | string | Source Mathlib lemma name. |
| `domain` | string | `number_theory`, `algebra`, `set_theory`, `combinatorics`, `analysis`, `order_theory`, `topology`. |
| `tier` | string | Difficulty tier `tier1`-`tier3`. |
| `family` | string | `canonical`, `syntactic`, `quantifier`, `passive`, `notation`, `order`, `unpack`, `equivalent`, `variation`. |
| `nl_question` | string | The exact TRUE/FALSE prompt. |
| `canonical_nl` | string | Canonical phrasing of the claim. |
| `ground_truth` | string | `"TRUE"` / `"FALSE"` (all rows are `"TRUE"` in current editions). |
| `verification_method` | string | `canonical` or `gpt4o_family_v2`. |
| `generation_seed` | int64 | Generation seed (`42`). |

**Is there a label or target associated with each instance?**
Yes -- `ground_truth`, the correct TRUE/FALSE verdict. In the released editions every label
is `"TRUE"`, because each paraphrase is a meaning-preserving rewording of a *true* Mathlib
theorem. FormInv therefore measures *consistency under rewording of true claims*; it is an
invariance probe rather than a balanced accuracy test.

**Is any information missing from individual instances?**
No fields are missing within a row; all twelve fields are populated for every instance. Some
paraphrase families are absent for some theorems by design (the transformation does not
apply), which lowers the per-family counts.

**Are relationships between individual instances made explicit?**
Yes. `theorem_id` links all paraphrases of one theorem, and `canonical_nl` /
`lean4_statement` / `mathlib_name` are shared across a theorem's paraphrase set. IG and SCR
are computed *within* a `theorem_id` group.

**Are there recommended data splits?**
There is no train/validation split -- FormInv is evaluation-only. The recommended strata are
the `all` / `canonical` / `surface` **configs** (each a single `test` split).

**Are there errors, sources of noise, or redundancies?**
Surface paraphrases are model-assisted (`gpt4o_family_v2`) and screened, but may carry
generator artifacts. Known issues and corrections are tracked in `ERRATA.md` in the GitHub
repository.

**Is the dataset self-contained or does it rely on external resources?**
Self-contained for evaluation. Base statements derive from Lean 4 / Mathlib; the provenance
(`mathlib_name`, `lean4_statement`) is embedded in each row, so no external download is
required to use the dataset.

**Does the dataset contain confidential / offensive / personal data?**
No. It contains only mathematical statements. There is no personal, sensitive, or offensive
content.

---

## 3. Collection Process

**How was the data associated with each instance acquired?**
Base theorems were selected from Lean 4 / Mathlib; the formal declaration and lemma name
were recorded directly (`lean4_statement`, `mathlib_name`). Canonical NL statements were
authored by the curators to faithfully express each formal statement
(`verification_method == "canonical"`). Surface paraphrases were then generated per family
via model-assisted rewriting and screened to preserve meaning
(`verification_method == "gpt4o_family_v2"`).

**What mechanisms or procedures were used to collect the data?**
A scripted generation pipeline (the FormInv builder in `forminv/`), seeded with
`generation_seed = 42` for reproducibility, plus manual authoring and screening of NL
statements. Labels follow deterministically from the truth of the source Mathlib theorem.

**If the dataset is a sample, what was the sampling strategy?**
Purposive selection of Mathlib theorems for clear NL rendering and coverage across the
domains and difficulty tiers listed above, rather than uniform random sampling.

**Who was involved in the data collection process?**
The authors (curation, canonical authoring, screening) plus model assistance for paraphrase
generation. A small expert pool contributed to the human-baseline / agreement study reported
in the paper.

**Over what timeframe was the data collected?**
During development of the FormInv `v3` editions (2026). See the paper and the GitHub
`CHANGELOG.md` for edition dates.

**Were any ethical review processes conducted?**
The dataset contains only mathematical statements and no personal data. Human-baseline study
procedures (instructions, consent) are described in the paper.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
Yes. Canonical NL was authored from the formal statement; surface paraphrases were generated
per family and screened for meaning preservation. Labels (`ground_truth`) were assigned from
the truth of the source theorem. For the Hugging Face release, the JSONL editions are
converted to Parquet and split into the `all` / `canonical` / `surface` configs by
`scripts/build_hf_dataset.py`; `ground_truth` is stored as a string column.

**Was the raw data saved in addition to the preprocessed data?**
Yes. The canonical JSONL editions (`data/generated/forminv_v3_50.jsonl`,
`forminv_v3_103.jsonl`) are retained in the GitHub repository, with `*.manifest.json`
(sha256 + row counts + seed) for integrity.

**Is the software used to preprocess/clean/label the data available?**
Yes -- the generation pipeline lives in `forminv/`, and the JSONL->Parquet build script is
`scripts/build_hf_dataset.py` in the GitHub repository (MIT-licensed).

---

## 5. Uses

**Has the dataset been used for any tasks already?**
Yes -- to benchmark semantic invariance (IG, SCR) across multiple LLM providers in the
FormInv paper.

**Is there a repository linking to papers or systems that use the dataset?**
Yes: the GitHub repository README maintains a static results table; the HF card links the
paper and leaderboard Space.

**What (other) tasks could the dataset be used for?**
Robustness analysis of mathematical-reasoning models, paraphrase-sensitivity studies,
calibration analysis, and as a template for constructing invariance probes in other formal
domains.

**Is there anything about the composition or collection that might impact future uses?**
Users must account for the **TRUE-only labels**: a model that answers TRUE unconditionally
will appear perfectly invariant. FormInv should be paired with a capability/accuracy
benchmark, and IG/SCR interpreted as *invariance* signals rather than correctness. Coverage
is small and English-only, and weighted toward `algebra`/`number_theory`.

**Are there tasks for which the dataset should not be used?**
It is a diagnostic research tool and must not be used for high-stakes decision-making, nor
as a standalone measure of mathematical correctness, nor as a fine-tuning corpus.

---

## 6. Distribution

**How will the dataset be distributed?**
Primarily via the Hugging Face Hub at `https://huggingface.co/datasets/forminv/forminv`
(Parquet configs + auto-generated Croissant metadata + dataset viewer). The canonical JSONL
editions are also available in the GitHub repository.

**When will the dataset be distributed?**
On release of the FormInv paper (arXiv:2605.29001); editions are versioned as Hub revisions
(`revision="v3.0"`).

**Will the dataset be distributed under a copyright/IP license or ToU?**
Yes -- **CC-BY-4.0** for the data. Base formal statements derive from **Lean 4 / Mathlib**,
licensed under **Apache-2.0**, which must be cited and respected; provenance is preserved in
`mathlib_name` / `lean4_statement`. The evaluation code is released separately under the
**MIT License**.

**Have any third parties imposed IP-based or other restrictions?**
Mathlib's Apache-2.0 license applies upstream to the formal statements; FormInv complies by
attributing Mathlib and preserving provenance.

**Do any export controls or regulatory restrictions apply?**
None known.

---

## 7. Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
Noel Thomas (`noel.thomas@mbzuai.ac.ae`), via the Hugging Face Hub and the GitHub repository.

**How can the owner/curator be contacted?**
Email `noel.thomas@mbzuai.ac.ae` or open an issue on the GitHub repository.

**Is there an erratum?**
Yes -- `ERRATA.md` in the GitHub repository tracks known issues and corrections.

**Will the dataset be updated?**
Yes. Updates (new theorems, paraphrase families, planned FALSE controls, corrections) are
released as new editions and tagged as Hub revisions; the canonical URL is stable across
editions. Per-item `generation_seed` and `*.manifest.json` checksums keep builds
reproducible.

**Will older versions continue to be supported/hosted?**
Yes. Prior editions remain accessible as Hub revisions (e.g. `v3_50`, `v3_103`), so results
remain reproducible.

**If others want to extend/contribute to the dataset, is there a mechanism?**
Yes. Paraphrase-family contributions and errata are welcome via GitHub pull requests /
issues, per `CONTRIBUTING.md`.

---

*This datasheet follows Gebru et al., "Datasheets for Datasets," Communications of the ACM,
2021.*
