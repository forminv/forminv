# Contributing to FormInv

Thanks for your interest in improving FormInv. This benchmark measures the
*semantic invariance* of LLM mathematical reasoning, so contributions that
broaden coverage or harden the measurement protocol are especially welcome.

## Ways to contribute

- **Report a data error** -- a mislabeled item, a paraphrase that changes meaning,
  or a canonical statement that misrepresents its Lean 4 / Mathlib source. Open an
  issue and add it to [`ERRATA.md`](ERRATA.md) in your PR.
- **Add a paraphrase family** -- a new meaning-preserving rewriting strategy
  (see `forminv/generators/families.py`). New families must be screened so the
  underlying truth value is preserved.
- **Add theorems** -- sampled from Lean 4 / Mathlib, with the `mathlib_name`
  recorded for provenance and a faithful canonical NL statement.
- **Improve the harness** -- new provider adapters, metrics, or analyses under
  `forminv/`.

## Development setup

This is a [uv](https://docs.astral.sh/uv/)-managed project.

```bash
# install uv: https://astral.sh/uv
git clone https://github.com/forminv/forminv && cd forminv
uv sync --all-groups          # runtime + dev tooling, installs forminv editable
uv sync --extra eval          # + provider SDKs (needed only to run live evals)
```

Run things with `uv run` (it auto-syncs first):

```bash
uv run pytest                 # tests
uv run ruff check .           # lint
uv run ruff format .          # format
uv run forminv eval --dataset data/generated/forminv_v3_50.jsonl --models all
```

## Code style

- Python >= 3.10, formatted and linted with **ruff** (config in `pyproject.toml`).
- **Google-style docstrings** (Args / Returns / Raises) on all public functions
  and classes.
- Keep functions focused; prefer pure functions for metrics so they are testable.
- Install the pre-commit hooks: `uv run pre-commit install`.

## Tests

- Add or update tests under `tests/` for any behavior change.
- Metric changes must include a numeric regression test.
- Provider parsing tests use **cached fixtures** -- never call a live API in CI.

## Pull request process

1. Branch from `main`, make focused commits.
2. Ensure `uv run ruff check .` and `uv run pytest` pass.
3. If you changed the dataset, bump the entry in [`CHANGELOG.md`](CHANGELOG.md)
   and record the data edition (see versioning in the README).
4. Describe *what* changed and *why* in the PR; link any related issue.

## Reporting data errata

Mathematical correctness is paramount. If you believe an item's label or a
paraphrase is wrong, please include the `id`, the `mathlib_name`, and a precise
explanation (ideally a counterexample). Confirmed errata are tracked in
`ERRATA.md` and corrected in the next data edition.

By contributing you agree that your contributions are licensed under the
project's MIT (code) and CC-BY-4.0 (data) licenses.
