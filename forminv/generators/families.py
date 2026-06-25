"""
FormInv Paraphrase Family Taxonomy -- 8 families (analogous to ChaosBench's 12 task families).

Each family tests a specific linguistic/mathematical invariance that LLMs should satisfy.
Ground truth for all paraphrases is the same as the base theorem (TRUE for valid Mathlib theorems).

FAMILY TAXONOMY:
  F1 -- surface_syntactic    : cosmetic rewrites (word order, sentence structure)
  F2 -- surface_quantifier   : quantifier word variation (all/every/any/for each)
  F3 -- active_passive       : active ↔ passive voice (where applicable)
  F4 -- formal_notation      : informal ↔ formal symbolic notation (m,n variables, 0%b)
  F5 -- connective_variation : logical connective lexical variation (iff/exactly when/iff=)
  F6 -- comparison_order     : inequality/comparison direction reversal (a≤b ↔ b≥a)
  F7 -- definitional_unpack  : unfold one mathematical definition (prime = 2 divisors)
  F8 -- domain_equivalent    : same claim in different mathematical register
"""

from enum import Enum


class Family(str, Enum):
    """The 8 FormInv paraphrase families.

    Each member names a linguistic or mathematical invariance that an
    equivalent restatement should preserve. See the module docstring and
    :data:`FAMILY_DESCRIPTIONS` for per-family descriptions.
    """

    F1_SURFACE_SYNTACTIC = "surface_syntactic"
    F2_SURFACE_QUANTIFIER = "surface_quantifier"
    F3_ACTIVE_PASSIVE = "active_passive"
    F4_FORMAL_NOTATION = "formal_notation"
    F5_CONNECTIVE_VARIATION = "connective_variation"
    F6_COMPARISON_ORDER = "comparison_order"
    F7_DEFINITIONAL_UNPACK = "definitional_unpack"
    F8_DOMAIN_EQUIVALENT = "domain_equivalent"


FAMILY_DESCRIPTIONS = {
    Family.F1_SURFACE_SYNTACTIC: "Cosmetic rewrite -- same words, different sentence structure",
    Family.F2_SURFACE_QUANTIFIER: "Quantifier word swap -- all/every/any/each/no exception",
    Family.F3_ACTIVE_PASSIVE: "Active/passive voice swap for relational claims",
    Family.F4_FORMAL_NOTATION: "Informal English ↔ symbolic notation (n%n, a|b, ∀n)",
    Family.F5_CONNECTIVE_VARIATION: "Logical connective swap (iff/exactly when/precisely if)",
    Family.F6_COMPARISON_ORDER: "Inequality direction reversal (a≤b ↔ b≥a ↔ ¬(a>b))",
    Family.F7_DEFINITIONAL_UNPACK: "Unfold one definition (prime ↔ exactly 2 divisors)",
    Family.F8_DOMAIN_EQUIVALENT: "Same structural claim in different domain/register",
}

# From gate check: which families cause failures in which models
KNOWN_FAILURE_PROFILES = {
    "gpt-4o": [
        Family.F3_ACTIVE_PASSIVE,
        Family.F6_COMPARISON_ORDER,
        Family.F5_CONNECTIVE_VARIATION,
    ],
    "claude-sonnet-4-6": [
        Family.F4_FORMAL_NOTATION,
        Family.F5_CONNECTIVE_VARIATION,
        Family.F2_SURFACE_QUANTIFIER,
    ],
}


# -- PARAPHRASE SYSTEM PROMPT (for GPT-4o generation) --------------------------

FAMILY_SYSTEM_PROMPT = """You are a mathematical linguist generating paraphrases of Lean4/Mathlib theorems.
Given a theorem, generate one paraphrase for each of 8 specified families.

RULES:
- ALL paraphrases must be mathematically equivalent to the original (same truth value: TRUE).
- Each paraphrase must be a yes/no or TRUE/FALSE question.
- Each must test ONLY the indicated linguistic transformation.
- Do NOT introduce new mathematical content or change scope.
- Do NOT add qualifications not in the original.

OUTPUT: JSON with keys matching the family names."""

FAMILY_USER_PROMPT = """Lean4 theorem: {lean4}
Canonical NL: {canonical}
Ground truth: TRUE (this is a valid Mathlib theorem)

Generate one paraphrase per family as a TRUE/FALSE question:

F1 (surface_syntactic): Rewrite with different sentence structure, same vocabulary.
F2 (surface_quantifier): Replace quantifier word (e.g., "every" → "any" or "for each" or "no...fails").
F3 (active_passive): Convert active voice to passive or vice versa (if applicable; skip if not relational).
F4 (formal_notation): Use symbolic notation instead of words (e.g., "n % n" instead of "n modulo n").
F5 (connective_variation): Change logical connective wording (iff → "exactly when" or "precisely when").
F6 (comparison_order): Reverse the comparison direction (a ≤ b → b ≥ a).
F7 (definitional_unpack): Replace one concept with its definition (prime → has exactly 2 positive divisors).
F8 (domain_equivalent): Express same claim using different but equivalent mathematical vocabulary.

Return JSON:
{{
  "surface_syntactic": "...",
  "surface_quantifier": "...",
  "active_passive": "... (or null if not applicable)",
  "formal_notation": "...",
  "connective_variation": "... (or null if no connective)",
  "comparison_order": "... (or null if no inequality)",
  "definitional_unpack": "...",
  "domain_equivalent": "..."
}}"""
