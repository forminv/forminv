"""
FormInv -- Generate publication-quality figures for the paper.

Produces:
  paper/fig_bimodal_ig.pdf   -- Bimodal IG distribution (Figure 1)
  paper/fig_family_failure.pdf -- Per-family failure rates (Figure 2)

Usage:
  cd /Users/noel.thomas/forminv
  .venv/bin/python scripts/generate_figures.py
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Paths --------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data/raw/response_cache"
DATASET_103 = ROOT / "data/generated/forminv_v3_103.jsonl"
FIG_DIR = ROOT / "figures"

USER_TEMPLATE = """{question}

Answer TRUE or FALSE."""


# -- Helpers ------------------------------------------------------------------
def cache_key(model: str, prompt: str) -> str:
    return hashlib.md5(f"{model}::{prompt}".encode()).hexdigest()[:16]


def compute_per_theorem_igs(dataset_path: Path, model: str) -> list[float]:
    """Per-theorem Invariance Gap values.

    Prefers the shipped precomputed values (``artifacts/bimodal_ig_values.json``)
    so figures regenerate without the raw response cache; falls back to computing
    from the cache when it is available.
    """
    precomp = ROOT / "artifacts/bimodal_ig_values.json"
    if precomp.exists():
        return json.loads(precomp.read_text())["per_theorem_ig"]
    with open(dataset_path) as f:
        items = [json.loads(l) for l in f]

    theorem_groups: dict[str, list] = {}
    for item in items:
        tid = item["theorem_id"]
        theorem_groups.setdefault(tid, []).append(item)

    ig_values = []
    for tid, thm_items in sorted(theorem_groups.items()):
        accs = []
        for item in thm_items:
            if "canonical" in item["id"]:
                continue
            prompt = USER_TEMPLATE.format(question=item["nl_question"])
            key = cache_key(model, prompt)
            fpath = CACHE_DIR / f"{key}.json"
            if fpath.exists():
                d = json.loads(fpath.read_text())
                if d["parsed"] is not None:
                    correct = float(d["parsed"] == item["ground_truth"])
                    accs.append(correct)
        if accs:
            ig_values.append(float(np.std(accs)))

    return ig_values


# -- Consistent colour palette ------------------------------------------------
COLORS = {
    "claude": "#E07B39",  # orange / Anthropic
    "gpt4o": "#3B82F6",  # blue / OpenAI
    "gpt4o_mini": "#10B981",  # green / OpenAI mini
}
FAMILY_ORDER = [
    "connective_variation",
    "comparison_order",
    "definitional_unpack",
    "active_passive",
    "formal_notation",
    "surface_syntactic",
    "domain_equivalent",
    "quantifier_variation",
]
FAMILY_LABELS = {
    "connective_variation": "F5 Connective\nvariation",
    "comparison_order": "F6 Comparison\norder",
    "definitional_unpack": "F7 Definitional\nunpack",
    "active_passive": "F3 Active/\npassive",
    "formal_notation": "F4 Formal\nnotation",
    "surface_syntactic": "F1 Surface\nsyntactic",
    "domain_equivalent": "F8 Domain\nequivalent",
    "quantifier_variation": "F2 Quantifier\nvariation",
}

# Per-family failure rates from the paper (Table in appendix).
# Claude numbers are from the 103-theorem dataset; GPT numbers from 50-theorem set.
FAMILY_FAILURES = {
    "connective_variation": {"claude": 19.2, "gpt4o": 22.2, "gpt4o_mini": 22.2},
    "comparison_order": {"claude": 8.3, "gpt4o": 0.0, "gpt4o_mini": 8.3},
    "definitional_unpack": {"claude": 7.8, "gpt4o": 10.0, "gpt4o_mini": 10.0},
    "active_passive": {"claude": 6.5, "gpt4o": 4.4, "gpt4o_mini": 4.4},
    "formal_notation": {"claude": 5.8, "gpt4o": 4.0, "gpt4o_mini": 2.0},
    "surface_syntactic": {"claude": 4.9, "gpt4o": 2.0, "gpt4o_mini": 10.0},
    "domain_equivalent": {"claude": 4.9, "gpt4o": 10.0, "gpt4o_mini": 4.0},
    "quantifier_variation": {"claude": 2.9, "gpt4o": 4.0, "gpt4o_mini": 2.0},
}


# -- Figure 1: Bimodal IG distribution ----------------------------------------
def fig_bimodal_ig(out_path: Path) -> None:
    ig_values = compute_per_theorem_igs(DATASET_103, "claude-sonnet-4-6")
    n_total = len(ig_values)
    n_zero = sum(1 for v in ig_values if v == 0.0)
    n_nonzero = n_total - n_zero

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    # Use fine bins; zero values form the spike at 0
    bins = np.linspace(0, 0.52, 28)
    counts, edges = np.histogram(ig_values, bins=bins)

    # Colour the zero-spike bar differently
    bar_colors = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == 0.0:
            bar_colors.append("#4B5563")  # dark grey for zero-IG
        else:
            bar_colors.append(COLORS["claude"])

    ax.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        color=bar_colors,
        align="edge",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # Annotations
    ax.annotate(
        f"79% IG = 0\n({n_zero} theorems)",
        xy=(0.008, n_zero),
        xytext=(0.06, n_zero - 2),
        fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        ha="left",
    )
    ax.annotate(
        f"21% IG > 0\n({n_nonzero} theorems)",
        xy=(0.35, 3),
        xytext=(0.20, 9),
        fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        ha="left",
    )

    ax.set_xlabel("Invariance Gap (IG)", fontsize=12)
    ax.set_ylabel("Number of theorems", fontsize=12)
    ax.set_title(
        "Distribution of per-theorem Invariance Gap\nClaude Sonnet 4.6 * 103-theorem FormInv dataset",
        fontsize=12,
    )
    ax.set_xlim(-0.01, 0.53)
    ax.set_ylim(0, None)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend patches
    zero_patch = mpatches.Patch(color="#4B5563", label="IG = 0  (fully invariant)")
    nonz_patch = mpatches.Patch(color=COLORS["claude"], label="IG > 0  (paraphrase-sensitive)")
    ax.legend(handles=[zero_patch, nonz_patch], fontsize=9.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# -- Figure 2: Per-family failure rates ---------------------------------------
def fig_family_failure(out_path: Path) -> None:
    models = ["claude", "gpt4o", "gpt4o_mini"]
    model_labels = {
        "claude": "Claude Sonnet 4.6",
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-mini",
    }
    model_colors = {k: COLORS[k] for k in models}

    families = FAMILY_ORDER
    n_families = len(families)
    n_models = len(models)

    x = np.arange(n_families)
    width = 0.22
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(10, 4.8))
    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    for i, (model, offset) in enumerate(zip(models, offsets)):
        values = [FAMILY_FAILURES[fam][model] for fam in families]
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=model_colors[model],
            label=model_labels[model],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        # Label bars above threshold
        for bar, val in zip(bars, values):
            if val >= 5.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="black",
                )

    # Highlight F5 as universally hardest
    ax.axvspan(-0.5, 0.5, alpha=0.08, color="#DC2626", zorder=0)
    ax.text(
        0,
        24.5,
        "Universal\nhardest",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#DC2626",
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [FAMILY_LABELS[fam] for fam in families],
        fontsize=9,
        ha="center",
    )
    ax.set_ylabel("Failure rate (%)", fontsize=12)
    ax.set_xlabel("Paraphrase family", fontsize=12)
    ax.set_title(
        "Per-family failure rates by model\n(Claude: 103-theorem dataset; GPT models: 50-theorem shared set)",
        fontsize=12,
    )
    ax.set_ylim(0, 29)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# -- Main ---------------------------------------------------------------------
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Figure 1: Bimodal IG distribution...")
    fig_bimodal_ig(FIG_DIR / "fig_bimodal_ig.pdf")

    print("Generating Figure 2: Per-family failure rates...")
    fig_family_failure(FIG_DIR / "fig_family_failure.pdf")

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
