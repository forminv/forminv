"""Evaluation subpackage for FormInv.

Provides the multi-provider LLM client (:func:`call_model`,
:func:`parse_label`) used to query models, along with the eval runner,
results analysis, and paraphrase-quality auditing modules.
"""

from .providers import call_model, parse_label
