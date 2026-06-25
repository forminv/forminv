"""
FormInvSelector -- Select the best model for a target reasoning task
based on per-family IG profile from FormInv evaluation.

Analogous to REIGN's RegimePlanner: uses per-family invariance profile
to recommend the model with minimum expected failure rate on the target task.

Usage:
    from forminv.selector import FormInvSelector

    selector = FormInvSelector.from_results("artifacts/all_models_results_v2.json")

    # For a task heavy on definitional reasoning (F7) and comparison direction (F6):
    best_model = selector.recommend(
        families=["unpack", "order"],   # F7 and F6
        top_k=3,
    )
    print(best_model)
    # -> [("deepseek/deepseek-chat", 0.032), ("openai/gpt-4o", 0.035), ...]

CLI:
    python -m forminv.selector --families unpack order --results artifacts/all_models_results_v2.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# Family name aliases (friendly -> internal)
FAMILY_ALIASES = {
    "syntactic": "syntactic",
    "f1": "syntactic",
    "quantifier": "quantifier",
    "f2": "quantifier",
    "passive": "passive",
    "active_passive": "passive",
    "f3": "passive",
    "notation": "notation",
    "formal_notation": "notation",
    "f4": "notation",
    "variation": "variation",
    "connective": "variation",
    "f5": "variation",
    "order": "order",
    "comparison": "order",
    "f6": "order",
    "unpack": "unpack",
    "definitional": "unpack",
    "f7": "unpack",
    "equivalent": "equivalent",
    "domain": "equivalent",
    "f8": "equivalent",
}

ALL_FAMILIES = [
    "syntactic",
    "quantifier",
    "passive",
    "notation",
    "variation",
    "order",
    "unpack",
    "equivalent",
]


@dataclass
class ModelProfile:
    """Per-model invariance profile extracted from FormInv eval results.

    Attributes:
        model_key: Provider-qualified model identifier (e.g. "openai/gpt-4o").
        mean_ig: Mean Invariance Gap across theorems.
        mean_accuracy: Mean accuracy across all paraphrases.
        scr: Strict Consistency Rate (fraction of theorems answered correctly
            on every paraphrase).
        family_failure_rates: Mapping of paraphrase family name to that model's
            failure rate on the family.
    """

    model_key: str
    mean_ig: float
    mean_accuracy: float
    scr: float
    family_failure_rates: dict[str, float]


class FormInvSelector:
    """Recommend the best model for a target reasoning task.

    Uses per-family Invariance Gap profiles to rank models by their expected
    failure rate on a task characterized by a set of paraphrase families.
    """

    def __init__(self, profiles: list[ModelProfile]):
        """Initialize the selector from a list of model profiles.

        Args:
            profiles: Model profiles to rank; keyed internally by ``model_key``.
        """
        self.profiles = {p.model_key: p for p in profiles}

    @classmethod
    def from_results(cls, results_path: str) -> FormInvSelector:
        """Build a selector from a FormInv eval results JSON file.

        Reads the ``per_model`` aggregates and constructs one
        :class:`ModelProfile` per model, skipping models whose mean accuracy is
        below 0.1 (treated as a coverage failure).

        Args:
            results_path: Path to the FormInv eval results JSON file.

        Returns:
            A :class:`FormInvSelector` populated with the eligible model profiles.
        """
        d = json.loads(Path(results_path).read_text())
        pm = d.get("per_model", {})
        profiles = []
        for mk, agg in pm.items():
            if agg.get("mean_accuracy", 0) < 0.1:
                continue  # skip models with coverage issues
            profiles.append(
                ModelProfile(
                    model_key=mk,
                    mean_ig=agg.get("mean_ig", 0.0),
                    mean_accuracy=agg.get("mean_accuracy", 0.0),
                    scr=agg.get("scr", 0.0),
                    family_failure_rates=agg.get("family_failure_rates", {}),
                )
            )
        return cls(profiles)

    def recommend(
        self,
        families: list[str],
        weights: dict[str, float] | None = None,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Recommend models for a task characterized by specific paraphrase families.

        Args:
            families: List of family names relevant to the target task
                (e.g., ["unpack", "order"] for definitional + comparison tasks).
            weights: Optional per-family weights (default: uniform).
            top_k: Number of top models to return.

        Returns:
            List of (model_key, expected_failure_rate) sorted ascending (best first).

        Raises:
            ValueError: If a family name is unknown, or if no families are given.
        """
        # Normalize family names
        norm_families = []
        for f in families:
            key = FAMILY_ALIASES.get(f.lower(), f.lower())
            if key in ALL_FAMILIES:
                norm_families.append(key)
            else:
                raise ValueError(f"Unknown family: {f!r}. Valid: {list(FAMILY_ALIASES.keys())}")

        if not norm_families:
            raise ValueError("Must specify at least one family.")

        # Uniform weights by default
        if weights is None:
            w = {f: 1.0 / len(norm_families) for f in norm_families}
        else:
            total = sum(weights.get(f, 1.0) for f in norm_families)
            w = {f: weights.get(f, 1.0) / total for f in norm_families}

        # Compute expected failure rate per model
        scores = []
        for mk, profile in self.profiles.items():
            efr = sum(w[f] * profile.family_failure_rates.get(f, profile.mean_ig) for f in norm_families)
            scores.append((mk, efr))

        return sorted(scores, key=lambda x: x[1])[:top_k]

    def family_ranking(self, family: str) -> list[tuple[str, float]]:
        """Rank models by failure rate on a single paraphrase family.

        Args:
            family: Family name or alias (e.g. "unpack" or "f7"); resolved via
                :data:`FAMILY_ALIASES`.

        Returns:
            List of (model_key, failure_rate) sorted ascending (best first).
            Models lacking a rate for the family fall back to their mean IG.
        """
        key = FAMILY_ALIASES.get(family.lower(), family.lower())
        scores = []
        for mk, profile in self.profiles.items():
            rate = profile.family_failure_rates.get(key, profile.mean_ig)
            scores.append((mk, rate))
        return sorted(scores, key=lambda x: x[1])

    def ranking_stability(self) -> dict[str, float]:
        """Compute Kendall tau between every pair of per-family model rankings.

        A low or negative tau between two families indicates that the relative
        model ordering reverses between them, signalling an unstable ranking.

        Returns:
            Mapping of "family_i,family_j" to the Kendall tau correlation
            between the two families' model rankings.
        """
        from scipy.stats import kendalltau

        results = {}
        for i, fi in enumerate(ALL_FAMILIES):
            ri = [mk for mk, _ in self.family_ranking(fi)]
            for j, fj in enumerate(ALL_FAMILIES):
                if j <= i:
                    continue
                rj = [mk for mk, _ in self.family_ranking(fj)]
                # Map to ranks
                ranks_i = [ri.index(m) for m in ri]
                ranks_j = [rj.index(m) for m in ri]
                tau, _ = kendalltau(ranks_i, ranks_j)
                results[f"{fi},{fj}"] = tau
        return results

    def print_recommendation(self, families: list[str], top_k: int = 5) -> None:
        """Print a human-readable model recommendation table.

        Args:
            families: Target paraphrase families characterizing the task.
            top_k: Number of top-ranked models to display.

        Returns:
            None.
        """
        recs = self.recommend(families, top_k=top_k)
        print(f"\nFormInvSelector -- Task: {families}")
        print(f"{'Model':<45} {'Expected Failure Rate':>20}")
        print("-" * 70)
        for mk, efr in recs:
            name = mk.split("/")[-1]
            print(f"{name:<45} {efr * 100:>19.1f}%")
        print()
        print(
            f"Recommendation: use {recs[0][0].split('/')[-1]} "
            f"(expected {recs[0][1] * 100:.1f}% failure on target families)"
        )


def main() -> None:
    """Command-line entry point for the standalone selector.

    Parses ``--families``, ``--results``, and ``--top-k``, prints the model
    recommendation table, and reports family-pair ranking reversals.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="FormInvSelector: recommend models for a target reasoning task")
    parser.add_argument(
        "--families",
        nargs="+",
        required=True,
        help="Paraphrase families for your task (e.g. unpack order variation)",
    )
    parser.add_argument(
        "--results",
        default="artifacts/all_models_results_v2.json",
        help="FormInv eval results JSON",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    selector = FormInvSelector.from_results(args.results)
    selector.print_recommendation(args.families, top_k=args.top_k)

    # Also show ranking stability
    print("Family-pair ranking stability (Kendall tau; negative = ranking reversal):")
    stability = selector.ranking_stability()
    reversals = [(k, v) for k, v in stability.items() if v < 0.3]
    for pair, tau in sorted(reversals, key=lambda x: x[1])[:5]:
        fi, fj = pair.split(",")
        print(f"  {fi:15} vs {fj:15}: tau={tau:.2f} {'<- REVERSAL' if tau < 0 else ''}")


if __name__ == "__main__":
    main()
