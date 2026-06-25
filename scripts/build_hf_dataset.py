#!/usr/bin/env python3
"""Build the FormInv Hugging Face dataset (JSONL -> Parquet, split into configs).

This script converts a FormInv edition JSONL file (e.g.
``data/generated/forminv_v3_103.jsonl``) into the Parquet layout that the
Hugging Face Hub expects, so that the dataset viewer renders and a valid
Croissant ``RecordSet`` is auto-generated (a NeurIPS Datasets & Benchmarks
submission gate).

HF layout produced
------------------
The output directory mirrors the dataset repo root. Each named **config**
(``all``, ``canonical``, ``surface``) gets its own ``test`` split written as a
single Parquet shard::

    <out>/
    |-- data/
    |   |-- all/test-00000-of-00001.parquet         # every row
    |   |-- canonical/test-00000-of-00001.parquet   # family == "canonical"
    |   `-- surface/test-00000-of-00001.parquet      # all non-canonical families

The ``config_name`` / ``data_files`` mapping that pairs with this layout lives
in the dataset card frontmatter (``hf_release/README.md``).

Configs (paraphrase strata)
---------------------------
* ``all``       -- the complete edition.
* ``canonical`` -- one canonical NL rendering per theorem (``family == "canonical"``).
* ``surface``   -- the meaning-preserving surface paraphrases (every other family:
  ``syntactic``, ``quantifier``, ``passive``, ``notation``, ``order``, ``unpack``,
  ``equivalent``, ``variation``).

Note: every row in the current editions is labelled ``ground_truth == "TRUE"``
(the editions are TRUE-only invariance probes), so there is no ``false_controls``
config. ``ground_truth`` is written as a plain string column (``"TRUE"`` /
``"FALSE"``) to keep the schema stable if FALSE controls are added later.

Usage
-----
Build the primary v3_103 edition::

    python scripts/build_hf_dataset.py \
        --input data/generated/forminv_v3_103.jsonl \
        --out hf_release

Build the smaller v3_50 edition into a separate revision tree::

    python scripts/build_hf_dataset.py \
        --input data/generated/forminv_v3_50.jsonl \
        --out hf_release_v3_50

Pushing to the Hub
------------------
After building, push the whole tree (card + LICENSE + ``data/``) with either
the ``huggingface-cli`` (now ``hf``) or the ``datasets`` library::

    # Option A: upload the prepared folder verbatim (card + Parquet)
    hf upload-large-folder <HF-ORG>/forminv hf_release --repo-type=dataset

    # Option B: load the configs and push them
    from datasets import load_dataset
    for cfg in ("all", "canonical", "surface"):
        ds = load_dataset("parquet",
                          data_files=f"hf_release/data/{cfg}/test-*.parquet")
        ds.push_to_hub("<HF-ORG>/forminv", config_name=cfg)

Tag the edition as a Hub revision (``revision="v3.0"``) after the first push so
the canonical URL is stable across editions.

This script only reads the input JSONL and writes Parquet; it performs no
network I/O and no Hub upload.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Columns in the canonical FormInv row schema, in card order.
EXPECTED_COLUMNS: list[str] = [
    "id",
    "theorem_id",
    "lean4_statement",
    "mathlib_name",
    "domain",
    "tier",
    "family",
    "nl_question",
    "canonical_nl",
    "ground_truth",
    "verification_method",
    "generation_seed",
]


def load_jsonl(input_path: Path) -> pd.DataFrame:
    """Loads a FormInv JSONL edition into a normalized DataFrame.

    Reads the newline-delimited JSON file, validates that every expected
    FormInv field is present, reorders the columns to the canonical card
    order, and casts ``ground_truth`` to a string column (``"TRUE"`` /
    ``"FALSE"``).

    Args:
        input_path: Path to a FormInv edition JSONL file (one JSON object per
            line), e.g. ``data/generated/forminv_v3_103.jsonl``.

    Returns:
        A DataFrame with exactly the columns in :data:`EXPECTED_COLUMNS`, with
        ``ground_truth`` stored as ``str``.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If any expected FormInv column is missing from the input.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    df = pd.read_json(input_path, lines=True)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input {input_path} is missing expected columns: {missing}. Found columns: {list(df.columns)}"
        )

    df = df[EXPECTED_COLUMNS].copy()
    # Store the label as a stable string; keeps schema fixed even if FALSE
    # controls are introduced in a later edition.
    df["ground_truth"] = df["ground_truth"].astype(str)
    return df


def split_configs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Splits a FormInv DataFrame into the HF configs.

    Args:
        df: A normalized FormInv DataFrame (see :func:`load_jsonl`).

    Returns:
        Mapping from config name to its DataFrame:
            * ``all``       -- every row.
            * ``canonical`` -- rows where ``family == "canonical"``.
            * ``surface``   -- rows where ``family != "canonical"``.
    """
    is_canonical = df["family"] == "canonical"
    return {
        "all": df,
        "canonical": df[is_canonical].reset_index(drop=True),
        "surface": df[~is_canonical].reset_index(drop=True),
    }


def write_configs(configs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Writes each config to ``<out_dir>/data/<config>/test-00000-of-00001.parquet``.

    Args:
        configs: Mapping from config name to DataFrame (see :func:`split_configs`).
        out_dir: Output root that mirrors the HF dataset repo. The ``data/``
            subtree is created if absent.

    Raises:
        ImportError: If ``pyarrow`` (the Parquet engine) is not installed.
    """
    for name, frame in configs.items():
        cfg_dir = out_dir / "data" / name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        shard = cfg_dir / "test-00000-of-00001.parquet"
        frame.to_parquet(shard, engine="pyarrow", index=False)
        print(f"  wrote {len(frame):>4d} rows -> {shard}")


def build(input_path: Path, out_dir: Path) -> None:
    """Runs the full JSONL -> Parquet build for one edition.

    Args:
        input_path: FormInv edition JSONL file.
        out_dir: Output root for the HF dataset layout.
    """
    df = load_jsonl(input_path)
    configs = split_configs(df)
    print(f"Loaded {len(df)} rows from {input_path}")
    write_configs(configs, out_dir)
    print(f"Done. HF layout written under {out_dir / 'data'}")


def main() -> None:
    """Parses CLI arguments and builds the dataset."""
    parser = argparse.ArgumentParser(
        description="Build the FormInv Hugging Face dataset (JSONL -> Parquet configs).",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a FormInv edition JSONL (e.g. data/generated/forminv_v3_103.jsonl).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for the HF dataset layout (data/<config>/*.parquet).",
    )
    args = parser.parse_args()
    build(args.input, args.out)


if __name__ == "__main__":
    main()
