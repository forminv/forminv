"""
Extract theorem statements from ntp-mathlib for FormInv Track B.
Target: ~500 theorems with natural-language names and clear statements.

Usage:
  .venv/bin/python scripts/extract_ntp_mathlib.py --n 500 --out data/generated/ntp_mathlib_candidates.jsonl
"""

import argparse
import json
import re
from pathlib import Path


def score_theorem(name: str, statement: str) -> float:
    """Score a theorem for suitability: higher = better for FormInv."""
    score = 0.0
    # Prefer theorems with human-readable names (not just symbols)
    if re.search(r"[a-z]{3,}", name.replace("_", "")):
        score += 1.0
    # Prefer short, clean statements (easier to paraphrase)
    if len(statement) < 200:
        score += 1.0
    if len(statement) < 100:
        score += 0.5
    # Prefer theorems that look like boolean/decidable propositions
    if any(kw in statement for kw in ["∀", "∃", "↔", "→", "Iff", "Even", "Odd", "Prime", "dvd"]):
        score += 1.0
    # Penalize theorems with complex type signatures
    if statement.count("→") > 3:
        score -= 1.0
    if "sorry" in statement.lower():
        score -= 5.0  # skip sorried proofs
    return score


def is_suitable(name: str, statement: str) -> bool:
    """Filter for theorems suitable as FormInv evaluation items."""
    # Must have a natural language-like name
    if not name or len(name) < 3:
        return False
    # Statement must be non-trivial
    if len(statement.strip()) < 10:
        return False
    # Skip tactics-only entries
    if statement.strip().startswith("by ") or statement.strip().startswith("exact "):
        return False
    return score_theorem(name, statement) > 0.5


def extract_theorem_name(item: dict) -> str:
    """
    Extract a short theorem name from an ntp-mathlib record.

    ntp-mathlib actual fields: decl (full statement), declId (qualified path),
    file_tag, state, srcUpToTactic, nextTactic, declUpToTactic.
    There is NO decl_nm field.  We extract the name from:
      1. decl_nm (legacy / other datasets)
      2. A regex on the decl string (theorem/lemma <name>)
      3. The last dotted component of declId before the line-number suffix
    """
    # Legacy or other dataset format
    if "decl_nm" in item:
        return item["decl_nm"]
    if "name" in item:
        return item["name"]

    # Parse name from full declaration text
    decl = item.get("decl", "")
    m = re.match(r"(?:@\[.*?\]\s*)?(?:theorem|lemma|def|abbrev)\s+(\S+)", decl.strip())
    if m:
        # Strip trailing parenthesis/colon that may be attached
        raw = m.group(1).rstrip(":({")
        return raw

    # No reliable name found -- return '' so is_suitable() will reject this record.
    # We intentionally do NOT fall back to declId module components (e.g. "Operations")
    # because those are Mathlib module names, not theorem names, and contaminate the output.
    return ""


def extract_from_local_jsonl(path: str, n: int) -> list[dict]:
    """Extract from a local JSONL file (ntp-mathlib or similar format)."""
    candidates = []
    seen_decls: set[str] = set()  # deduplicate by statement
    field_path_logged = False

    with open(path) as f:
        for i, line in enumerate(f):
            item = json.loads(line)

            if not field_path_logged:
                print(f"[extract] Sample keys: {list(item.keys())[:8]}")
                field_path_logged = True

            name = extract_theorem_name(item)
            statement = item.get("decl", item.get("statement", ""))

            # Deduplicate: ntp-mathlib repeats the same decl for each tactic step
            stmt_key = statement.strip()[:200]
            if stmt_key in seen_decls:
                continue
            seen_decls.add(stmt_key)

            if is_suitable(name, statement):
                score = score_theorem(name, statement)
                candidates.append(
                    {
                        "mathlib_name": name,
                        "lean4_statement": statement,
                        "score": score,
                        "source": "ntp-mathlib",
                    }
                )

    print(
        f"[extract] Scanned {len(seen_decls)} unique decls, {len(candidates)} passed suitability filter (before top-n cut)"
    )
    # Sort by score and return top n
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument(
        "--input",
        default="data/raw/ntp_mathlib_sample.jsonl",
        help="Local ntp-mathlib JSONL file (download separately)",
    )
    parser.add_argument("--out", default="data/generated/ntp_mathlib_candidates.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Download from: huggingface.co/datasets/l3lab/ntp-mathlib")
        print("Or run: .venv/bin/python scripts/download_ntp_mathlib.py")
        print()
        print("Creating a synthetic placeholder with 20 hand-crafted theorem stubs for testing...")
        stubs = [
            {
                "mathlib_name": "nat_even_add_odd",
                "lean4_statement": "theorem nat_even_add_odd (n : ℕ) (h : Even n) (k : ℕ) (hk : Odd k) : Odd (n + k)",
                "score": 2.0,
                "source": "stub",
            },
            {
                "mathlib_name": "int_mul_comm",
                "lean4_statement": "theorem int_mul_comm (a b : ℤ) : a * b = b * a",
                "score": 2.5,
                "source": "stub",
            },
            {
                "mathlib_name": "real_abs_nonneg",
                "lean4_statement": "theorem real_abs_nonneg (x : ℝ) : 0 ≤ |x|",
                "score": 2.5,
                "source": "stub",
            },
            {
                "mathlib_name": "nat_prime_two_le",
                "lean4_statement": "theorem nat_prime_two_le {p : ℕ} (hp : p.Prime) : 2 ≤ p",
                "score": 3.0,
                "source": "stub",
            },
            {
                "mathlib_name": "int_even_iff_dvd_two",
                "lean4_statement": "theorem int_even_iff_dvd_two (n : ℤ) : Even n ↔ 2 ∣ n",
                "score": 3.0,
                "source": "stub",
            },
            {
                "mathlib_name": "nat_dvd_antisymm",
                "lean4_statement": "theorem Nat.dvd_antisymm : ∀ {m n : ℕ}, m ∣ n → n ∣ m → m = n",
                "score": 2.5,
                "source": "stub",
            },
            {
                "mathlib_name": "nat_odd_iff",
                "lean4_statement": "theorem Nat.odd_iff {n : ℕ} : Odd n ↔ n % 2 = 1",
                "score": 3.0,
                "source": "stub",
            },
            {
                "mathlib_name": "int_abs_nonneg",
                "lean4_statement": "theorem Int.natAbs_pos {n : ℤ} (h : n ≠ 0) : 0 < n.natAbs",
                "score": 2.5,
                "source": "stub",
            },
            {
                "mathlib_name": "nat_succ_pos",
                "lean4_statement": "theorem Nat.succ_pos : ∀ (n : ℕ), 0 < n.succ",
                "score": 2.5,
                "source": "stub",
            },
            {
                "mathlib_name": "nat_pos_of_prime",
                "lean4_statement": "theorem Nat.Prime.pos {p : ℕ} (hp : p.Prime) : 0 < p",
                "score": 3.0,
                "source": "stub",
            },
            {
                "mathlib_name": "real_sq_nonneg",
                "lean4_statement": "theorem sq_nonneg (a : ℝ) : 0 ≤ a ^ 2",
                "score": 2.5,
                "source": "stub",
            },
            {
                "mathlib_name": "nat_even_two_mul",
                "lean4_statement": "theorem nat_even_two_mul (k : ℕ) : Even (2 * k)",
                "score": 2.5,
                "source": "stub",
            },
            {
                "mathlib_name": "prime_def_lt_dvd",
                "lean4_statement": "theorem Nat.prime_def_lt_dvd {p : ℕ} : p.Prime ↔ 2 ≤ p ∧ ∀ m, 2 ≤ m → m < p → ¬m ∣ p",
                "score": 3.5,
                "source": "stub",
            },
            {
                "mathlib_name": "nat_dvd_refl",
                "lean4_statement": "theorem Nat.dvd_refl (n : ℕ) : n ∣ n",
                "score": 2.0,
                "source": "stub",
            },
            {
                "mathlib_name": "int_mul_zero",
                "lean4_statement": "theorem mul_zero (a : ℤ) : a * 0 = 0",
                "score": 2.0,
                "source": "stub",
            },
            {
                "mathlib_name": "nat_add_comm",
                "lean4_statement": "theorem Nat.add_comm (n m : ℕ) : n + m = m + n",
                "score": 2.0,
                "source": "stub",
            },
            {
                "mathlib_name": "real_lt_irrefl",
                "lean4_statement": "theorem lt_irrefl (a : ℝ) : ¬a < a",
                "score": 2.0,
                "source": "stub",
            },
            {
                "mathlib_name": "nat_prime_odd_or_eq_two",
                "lean4_statement": "theorem Nat.Prime.eq_two_or_odd {p : ℕ} (hp : p.Prime) : p = 2 ∨ Odd p",
                "score": 3.5,
                "source": "stub",
            },
            {
                "mathlib_name": "int_dvd_zero",
                "lean4_statement": "theorem dvd_zero (a : ℤ) : a ∣ 0",
                "score": 2.0,
                "source": "stub",
            },
            {
                "mathlib_name": "nat_even_add_even",
                "lean4_statement": "theorem Even.add_even {m n : ℕ} (hm : Even m) (hn : Even n) : Even (m + n)",
                "score": 3.0,
                "source": "stub",
            },
        ]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for stub in stubs:
                f.write(json.dumps(stub) + "\n")
        print(f"Created {len(stubs)} stub theorems at {args.out}")
        print("\nTop 10 candidates (stubs):")
        for c in stubs[:10]:
            print(f"  {c['mathlib_name'][:50]:<50} score={c['score']:.1f}")
        return

    candidates = extract_from_local_jsonl(args.input, args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    print(f"Extracted {len(candidates)} candidate theorems to {args.out}")

    # Print domain summary
    print("\nTop 10 candidates:")
    for c in candidates[:10]:
        print(f"  {c['mathlib_name'][:50]:<50} score={c['score']:.1f}")


if __name__ == "__main__":
    main()
