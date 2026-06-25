"""Download ntp-mathlib sample for Track B scaling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("Downloading ntp-mathlib sample (5000 items)...")
    try:
        from datasets import load_dataset

        ds = load_dataset("l3lab/ntp-mathlib", split="train[:5000]")
        out = Path("data/raw/ntp_mathlib_sample.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_json(str(out))
        print(f"Downloaded {len(ds)} items to {out}")
        # Show one record so field names can be verified
        sample = ds[0]
        print(f"\nSample record keys: {list(sample.keys())}")
        print(f"Sample decl_nm: {sample.get('decl_nm', sample.get('name', '<not found>'))[:80]}")
        print(f"Sample decl:    {sample.get('decl', sample.get('statement', '<not found>'))[:120]}")
    except ImportError:
        print("Install: .venv/bin/pip install datasets")
    except Exception as e:
        print(f"Failed: {e}")
        print("Manual: go to huggingface.co/datasets/l3lab/ntp-mathlib and download train split")


if __name__ == "__main__":
    main()
