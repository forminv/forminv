"""FormInv: Formalization-Invariance Diagnostic for LLM Mathematical Understanding.

FormInv measures whether large language models give consistent answers to
mathematically equivalent restatements of the same theorem. It pairs curated
Mathlib theorems with families of natural-language paraphrases and reports the
Invariance Gap (IG) and Strict Consistency Rate (SCR) per model.

The package is organized into:
    - ``schemas``: core dataclasses for theorems, paraphrases, and results.
    - ``generators``: theorem loading and paraphrase generation.
    - ``eval``: multi-provider model evaluation, analysis, and auditing.
    - ``metrics``: Invariance Gap and consistency metrics.
    - ``selector``: model recommendation from per-family IG profiles.
    - ``cli``: the ``forminv`` command-line entry point.
"""

__version__ = "3.0.0"
