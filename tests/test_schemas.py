"""Construction and enum tests for ``forminv.schemas``."""

from forminv.schemas import (
    BaseTheorem,
    DomainTier,
    EquivLevel,
    ModelPrediction,
    Paraphrase,
)


def test_equiv_level_values():
    assert EquivLevel.SURFACE.value == "surface"
    assert EquivLevel.DEFINITIONAL.value == "definitional"


def test_domain_tier_values():
    assert DomainTier.T1.value == "tier1"
    assert DomainTier.T4.value == "tier4"


def test_base_theorem_defaults():
    thm = BaseTheorem(
        theorem_id="thm_0",
        lean4_statement="theorem foo : True",
        canonical_nl="A trivially true statement.",
        ground_truth=True,
        domain="number_theory",
        tier=DomainTier.T1,
        mathlib_name="Foo.bar",
    )
    assert thm.seed == 42
    assert thm.cas_verifiable is False
    assert thm.cas_proof is None


def test_paraphrase_defaults():
    p = Paraphrase(
        paraphrase_id="thm_0_canonical",
        theorem_id="thm_0",
        level=EquivLevel.SURFACE,
        nl_question="Is this true?",
        ground_truth=True,
    )
    assert p.verification_method == "human"
    assert p.equivalence_verified is False


def test_model_prediction_roundtrip():
    pred = ModelPrediction(
        pred_id="p0__m",
        paraphrase_id="p0",
        theorem_id="t0",
        model="m",
        provider="mock",
        raw_response="TRUE",
        parsed_label=True,
        outcome="TRUE",
        correct=True,
        latency_s=0.01,
    )
    assert pred.parsed_label is True
    assert pred.outcome == "TRUE"
