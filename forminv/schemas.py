"""Core data schemas for FormInv.

Defines the dataclasses and enumerations shared across the package: base
theorems, their paraphrases, individual model predictions, per-theorem
Invariance Gap results, and the top-level run manifest.
"""

from dataclasses import dataclass
from enum import Enum


class EquivLevel(str, Enum):
    """Equivalence level of a paraphrase relative to its base theorem.

    Attributes:
        SURFACE: Same vocabulary, different word order or phrasing.
        DEFINITIONAL: Different vocabulary, equivalent by definition.
    """

    SURFACE = "surface"  # Same words, different order/phrasing
    DEFINITIONAL = "definitional"  # Different vocab, equiv by definition


class DomainTier(str, Enum):
    """Difficulty tier of a theorem's mathematical domain.

    Attributes:
        T1: Freshman level (primality, divisibility, modular, basic algebra).
        T2: Sophomore level (polynomial, group theory basics, set theory).
        T3: Junior level (topology definitions, ring theory, real analysis).
        T4: Senior level (abstract algebra, measure theory, number theory).
    """

    T1 = "tier1"  # Freshman: primality, divisibility, modular, basic algebra
    T2 = "tier2"  # Sophomore: polynomial, group theory basics, set theory
    T3 = "tier3"  # Junior: topology defs, ring theory, real analysis
    T4 = "tier4"  # Senior: abstract algebra, measure theory, number theory


@dataclass
class BaseTheorem:
    """A Mathlib theorem statement with canonical NL phrasing."""

    theorem_id: str
    lean4_statement: str  # Full Lean4 theorem declaration
    canonical_nl: str  # Canonical natural-language statement
    ground_truth: bool  # Always True for valid theorems
    domain: str  # e.g. "number_theory", "algebra"
    tier: DomainTier
    mathlib_name: str  # e.g. "Nat.Prime.eq_one_or_self"
    cas_verifiable: bool = False  # Whether SymPy can verify the claim
    cas_proof: str | None = None
    seed: int = 42


@dataclass
class Paraphrase:
    """A paraphrase of a theorem in a specific NL question form."""

    paraphrase_id: str  # e.g. "natprime_surf_1"
    theorem_id: str
    level: EquivLevel
    nl_question: str  # The actual question asked to the model
    ground_truth: bool
    lean4_equivalent: str | None = None  # Lean4 reformulation (for verification)
    equivalence_verified: bool = False  # Whether Lean4 checked equivalence
    verification_method: str = "human"  # "lean4", "sympy", "human"


@dataclass
class ModelPrediction:
    """A single model prediction on a paraphrase."""

    pred_id: str
    paraphrase_id: str
    theorem_id: str
    model: str
    provider: str
    raw_response: str
    parsed_label: bool | None  # None = invalid
    outcome: str  # "TRUE", "FALSE", "INVALID"
    correct: bool | None
    latency_s: float


@dataclass
class InvarianceGapResult:
    """Invariance Gap computed for one base theorem + one model."""

    theorem_id: str
    model: str
    n_paraphrases: int
    accuracy_per_paraphrase: list[float]  # acc for each paraphrase
    mean_accuracy: float
    std_accuracy: float  # This IS the IG (variance-based)
    all_correct: bool  # SCR: all paraphrases correct?
    baseline_variance: float  # Within-canonical noise floor


@dataclass
class RunManifest:
    """Top-level run manifest."""

    run_id: str
    dataset_id: str
    model: str
    provider: str
    n_theorems: int
    n_paraphrases: int
    total_calls: int
    coverage: float
    mean_ig: float
    baseline_variance: float
    ig_exceeds_baseline: bool
    timestamp: str
