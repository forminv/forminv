"""Invariance Gap and related metrics for FormInv -- v2 with per-family breakdown.

Provides the core FormInv measurements: the per-theorem Invariance Gap
(:func:`compute_ig`), its per-family breakdown (:func:`compute_ig_by_family`),
cross-theorem aggregation including the Strict Consistency Rate
(:func:`aggregate_ig`), and cross-model disagreement detection
(:func:`compute_cross_model_disagreement`).
"""

from __future__ import annotations

import numpy as np

from forminv.generators.families import Family
from forminv.schemas import InvarianceGapResult, ModelPrediction, Paraphrase


def compute_ig(
    theorem_id: str,
    model: str,
    predictions: list[ModelPrediction],
    paraphrases: list[Paraphrase],
    baseline_variance: float = 0.0,
) -> InvarianceGapResult:
    """Compute the Invariance Gap for one theorem and one model.

    Selects the model's predictions over the theorem's paraphrases and measures
    the spread (standard deviation) of per-paraphrase accuracy, which is the
    variance-based Invariance Gap.

    Args:
        theorem_id: The theorem whose paraphrases are scored.
        model: The model whose predictions are scored.
        predictions: All model predictions to filter by theorem and paraphrase.
        paraphrases: The theorem's paraphrases (used to select prediction IDs).
        baseline_variance: Within-canonical noise floor to record alongside the
            result.

    Returns:
        An :class:`~forminv.schemas.InvarianceGapResult`; when no scorable
        predictions exist it is a zero-filled result with ``all_correct`` False.
    """
    para_ids = {p.paraphrase_id for p in paraphrases if p.theorem_id == theorem_id}
    preds = [p for p in predictions if p.paraphrase_id in para_ids]

    acc_by_para: dict[str, float] = {}
    for pred in preds:
        if pred.correct is not None:
            acc_by_para[pred.paraphrase_id] = float(pred.correct)

    acc_per_para = list(acc_by_para.values())
    if not acc_per_para:
        return InvarianceGapResult(
            theorem_id=theorem_id,
            model=model,
            n_paraphrases=0,
            accuracy_per_paraphrase=[],
            mean_accuracy=0.0,
            std_accuracy=0.0,
            all_correct=False,
            baseline_variance=baseline_variance,
        )

    return InvarianceGapResult(
        theorem_id=theorem_id,
        model=model,
        n_paraphrases=len(acc_per_para),
        accuracy_per_paraphrase=acc_per_para,
        mean_accuracy=float(np.mean(acc_per_para)),
        std_accuracy=float(np.std(acc_per_para)),
        all_correct=all(a == 1.0 for a in acc_per_para),
        baseline_variance=baseline_variance,
    )


def compute_ig_by_family(
    theorem_id: str, model: str, predictions: list[ModelPrediction], paraphrases: list[Paraphrase]
) -> dict[str, float]:
    """Compute the Invariance Gap broken down by paraphrase family.

    For each non-canonical paraphrase of the theorem, records whether the model
    answered it incorrectly, keyed by the family inferred from the paraphrase ID.

    Args:
        theorem_id: The theorem whose paraphrases are scored.
        model: The model whose predictions are scored.
        predictions: All model predictions to match by paraphrase ID and model.
        paraphrases: The theorem's paraphrases to iterate over.

    Returns:
        A dict mapping family name to 0.0 (correct) or 1.0 (incorrect/failure).
    """
    family_results = {}
    for para in paraphrases:
        if para.theorem_id != theorem_id or "canonical" in para.paraphrase_id:
            continue
        pred = next(
            (p for p in predictions if p.paraphrase_id == para.paraphrase_id and p.model == model),
            None,
        )
        if pred and pred.correct is not None:
            # Extract family from paraphrase_id
            family_key = None
            for f in Family:
                if f.value in para.paraphrase_id:
                    family_key = f.value
                    break
            if family_key:
                family_results[family_key] = float(not pred.correct)  # 1.0 = failure
    return family_results


def aggregate_ig(ig_results: list[InvarianceGapResult]) -> dict:
    """Aggregate per-theorem Invariance Gap results into summary statistics.

    Args:
        ig_results: Per-theorem/model results to aggregate.

    Returns:
        A summary dict with mean/std/max IG, mean accuracy, the Strict
        Consistency Rate (``scr``), the mean baseline variance, a flag for
        whether mean IG exceeds twice the baseline, the theorem count, and the
        fraction of results with IG above 0.10 and 0.20. Empty if no results.
    """
    if not ig_results:
        return {}

    all_ig = [r.std_accuracy for r in ig_results]
    all_mean_acc = [r.mean_accuracy for r in ig_results]
    all_baseline = [r.baseline_variance for r in ig_results]
    scr_list = [r.all_correct for r in ig_results]
    mean_baseline = float(np.mean(all_baseline)) if all_baseline else 0.0

    return {
        "mean_ig": float(np.mean(all_ig)),
        "std_ig": float(np.std(all_ig)),
        "max_ig": float(np.max(all_ig)),
        "mean_accuracy": float(np.mean(all_mean_acc)),
        "scr": float(np.mean(scr_list)),
        "mean_baseline_variance": mean_baseline,
        "ig_exceeds_2x_baseline": float(np.mean(all_ig)) > 2 * mean_baseline,
        "n_theorems": len(ig_results),
        "pct_high_ig_10": float(np.mean([ig > 0.10 for ig in all_ig])),
        "pct_high_ig_20": float(np.mean([ig > 0.20 for ig in all_ig])),
    }


def compute_cross_model_disagreement(
    predictions_by_model: dict[str, list[ModelPrediction]], paraphrases: list[Paraphrase]
) -> list[dict]:
    """Find paraphrases on which two models produced different labels.

    Args:
        predictions_by_model: Mapping of model name to that model's predictions.
        paraphrases: Paraphrases to check for cross-model disagreement.

    Returns:
        One dict per disagreeing model pair and paraphrase, with keys
        ``paraphrase_id``, ``theorem_id``, ``model_a``, ``label_a``,
        ``model_b``, ``label_b``, and ``ground_truth``. Empty if fewer than two
        models are provided.
    """
    disagreements = []
    models = list(predictions_by_model.keys())
    if len(models) < 2:
        return []

    for para in paraphrases:
        labels = {}
        for m, preds in predictions_by_model.items():
            pred = next((p for p in preds if p.paraphrase_id == para.paraphrase_id), None)
            if pred:
                labels[m] = pred.outcome

        # Check if any pair disagrees
        model_list = list(labels.keys())
        for i in range(len(model_list)):
            for j in range(i + 1, len(model_list)):
                ma, mb = model_list[i], model_list[j]
                if labels.get(ma) != labels.get(mb):
                    disagreements.append(
                        {
                            "paraphrase_id": para.paraphrase_id,
                            "theorem_id": para.theorem_id,
                            "model_a": ma,
                            "label_a": labels.get(ma),
                            "model_b": mb,
                            "label_b": labels.get(mb),
                            "ground_truth": "TRUE" if para.ground_truth else "FALSE",
                        }
                    )
    return disagreements
