# Changelog

All notable changes to FormInv are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.0] - 2026-06-23

### Added
- **v3 dataset**: `forminv_v3_50.jsonl` (366 items, 50 theorems) and
  `forminv_v3_103.jsonl` (760 items, 103 theorems), each theorem rendered as a
  canonical NL statement plus surface paraphrase families, every item labeled
  TRUE/FALSE with Lean 4 / Mathlib provenance.
- **Conditional Inconsistency Rate (CIR)**: a difficulty-controlled invariance
  metric -- among theorems a model answers correctly in canonical form, the
  fraction with at least one meaning-preserving paraphrase that flips the verdict.
- **Multi-sample CIR** (n=8, T=0.7): separates *systematic* invariance failures
  from *stochastic* (temperature) noise.
- **SCR mechanical-floor decomposition**: a permutation null that quantifies how
  much of the cross-model SCR spread is an arithmetic consequence of accuracy vs.
  genuine within-theorem failure clustering.
- **Balanced TRUE/FALSE controls** with subtle (single meaning-changing edit)
  FALSE siblings, defusing the all-TRUE confound.
- Multi-provider evaluation harness (OpenAI, Anthropic, DeepSeek, Gemini).
- uv-managed packaging: PEP 735 dependency groups, an `eval` extra for provider
  SDKs, committed lockfile for reproducible installs.

### Changed
- Headline metric reframed from raw SCR to CIR after showing the raw SCR spread
  is largely a mechanical (p^k) consequence of accuracy.

### Fixed
- Corrected the Gemini SDK dependency (`google-generativeai` -> `google-genai`).

[Unreleased]: https://github.com/forminv/forminv/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/forminv/forminv/releases/tag/v3.0.0
