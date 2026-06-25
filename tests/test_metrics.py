"""Numeric regression tests for ``forminv.metrics``.

Covers the Invariance Gap (IG / std-based), the Self-Consistency Rate (SCR /
``all_correct``), aggregation, and cross-model disagreement detection. All inputs
are synthetic fixtures.
"""

from forminv.metrics import (
    aggregate_ig,
    compute_cross_model_disagreement,
    compute_ig,
)
from forminv.schemas import EquivLevel, ModelPrediction, Paraphrase


def _para(pid, tid="t0", gt=True):
    return Paraphrase(
        paraphrase_id=pid,
        theorem_id=tid,
        level=EquivLevel.SURFACE,
        nl_question="q",
        ground_truth=gt,
    )


def _pred(pid, correct, tid="t0", model="m", outcome="TRUE"):
    return ModelPrediction(
        pred_id=f"{pid}__{model}",
        paraphrase_id=pid,
        theorem_id=tid,
        model=model,
        provider="mock",
        raw_response=outcome,
        parsed_label=(outcome == "TRUE"),
        outcome=outcome,
        correct=correct,
        latency_s=0.0,
    )


def test_compute_ig_all_correct_is_invariant():
    paras = [_para(f"t0_p{i}") for i in range(3)]
    preds = [_pred(f"t0_p{i}", correct=True) for i in range(3)]
    r = compute_ig("t0", "m", preds, paras)
    assert r.n_paraphrases == 3
    assert r.mean_accuracy == 1.0
    assert r.std_accuracy == 0.0  # IG == 0 when perfectly invariant
    assert r.all_correct is True  # SCR hit


def test_compute_ig_mixed_has_positive_gap():
    paras = [_para(f"t0_p{i}") for i in range(3)]
    preds = [_pred("t0_p0", True), _pred("t0_p1", True), _pred("t0_p2", False)]
    r = compute_ig("t0", "m", preds, paras)
    assert r.n_paraphrases == 3
    assert abs(r.mean_accuracy - 2 / 3) < 1e-9
    assert r.std_accuracy > 0.0  # IG > 0 when phrasing changes the verdict
    assert r.all_correct is False


def test_compute_ig_no_predictions_is_empty():
    paras = [_para("t0_p0")]
    r = compute_ig("t0", "m", [], paras)
    assert r.n_paraphrases == 0
    assert r.all_correct is False


def test_aggregate_ig_scr_is_fraction_all_correct():
    paras = [_para(f"t0_p{i}") for i in range(2)]
    invariant = compute_ig("t0", "m", [_pred("t0_p0", True), _pred("t0_p1", True)], paras)

    paras_b = [_para(f"t1_p{i}", tid="t1") for i in range(2)]
    brittle = compute_ig(
        "t1",
        "m",
        [_pred("t1_p0", True, tid="t1"), _pred("t1_p1", False, tid="t1")],
        paras_b,
    )
    agg = aggregate_ig([invariant, brittle])
    assert agg["n_theorems"] == 2
    assert agg["scr"] == 0.5  # one of two theorems fully consistent
    assert agg["mean_ig"] >= 0.0


def test_aggregate_ig_empty_returns_empty_dict():
    assert aggregate_ig([]) == {}


def test_cross_model_disagreement_detects_flip():
    para = _para("t0_p0")
    preds_by_model = {
        "a": [_pred("t0_p0", True, model="a", outcome="TRUE")],
        "b": [_pred("t0_p0", False, model="b", outcome="FALSE")],
    }
    dis = compute_cross_model_disagreement(preds_by_model, [para])
    assert len(dis) == 1
    assert dis[0]["paraphrase_id"] == "t0_p0"


def test_cross_model_agreement_yields_nothing():
    para = _para("t0_p0")
    preds_by_model = {
        "a": [_pred("t0_p0", True, model="a", outcome="TRUE")],
        "b": [_pred("t0_p0", True, model="b", outcome="TRUE")],
    }
    assert compute_cross_model_disagreement(preds_by_model, [para]) == []
