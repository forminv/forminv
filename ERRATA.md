# Errata

Known data issues and their resolutions. We take mathematical correctness
seriously; if you find a mislabeled item or a meaning-changing paraphrase, please
open an issue (see [CONTRIBUTING.md](CONTRIBUTING.md)) and it will be triaged here.

Corrections are applied in the next data edition; the affected edition and the
fixing edition are both recorded so results remain interpretable.

| Date | Item `id` | Edition | Issue | Resolution |
|------|-----------|---------|-------|------------|
| --    | --         | --       | No errata reported to date. | -- |

## How items are verified

- **Canonical items** inherit their label directly from the source Lean 4 /
  Mathlib statement (`mathlib_name`).
- **Paraphrases** are screened to preserve the underlying truth value
  (`verification_method` records how).
- **FALSE controls** carry an explicit counterexample; numeric ones are
  machine-checked.

Reported issues that turn out to be correct-as-labeled are recorded here too, with
the rationale, to avoid repeat reports.
