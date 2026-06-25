"""Convert ntp-mathlib candidates to FormInv theorem format.

Generates canonical_nl (natural language) from the Lean4 statement using
gpt-4o-mini, assigns domain from module name, and writes JSONL output
compatible with BaseTheorem and downstream paraphrase generation.

Usage:
  # First 30 (smoke test):
  .venv/bin/python scripts/convert_ntp_to_formInv.py --n 30
  # Full 100:
  .venv/bin/python scripts/convert_ntp_to_formInv.py --n 100
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Domain map -- order matters: more specific keys first to avoid short-string
# false matches (e.g. 'Ring' matching 'RingTheory').
DOMAIN_MAP = [
    ("MeasureTheory", "analysis"),
    ("CategoryTheory", "category_theory"),
    ("ModelTheory", "logic"),
    ("GroupTheory", "algebra"),
    ("FieldTheory", "algebra"),
    ("RingTheory", "algebra"),
    ("LinearAlgebra", "algebra"),
    ("AlgebraicGeometry", "algebra"),
    ("NumberTheory", "number_theory"),
    ("Combinatorics", "combinatorics"),
    ("Topology", "topology"),
    ("Analysis", "analysis"),
    ("Algebra", "algebra"),
    ("Geometry", "geometry"),
    ("Logic", "logic"),
    ("Computability", "logic"),
    # Data.* submodules
    ("Nat", "number_theory"),
    ("Int", "number_theory"),
    ("Prime", "number_theory"),
    ("Real", "analysis"),
    ("Complex", "analysis"),
    ("Metric", "topology"),
    ("Set", "set_theory"),
    ("Finset", "combinatorics"),
    ("Multiset", "combinatorics"),
    ("List", "combinatorics"),
    ("Fin", "combinatorics"),
    ("Matrix", "algebra"),
    ("Polynomial", "algebra"),
    ("Order", "order_theory"),
    ("Lattice", "order_theory"),
    ("Ring", "algebra"),
    ("Group", "algebra"),
    ("Field", "algebra"),
]


def assign_domain(file_tag: str, name: str) -> str:
    """Assign a FormInv domain from file_tag (preferred) and theorem name."""
    # Prefer file_tag-based assignment
    tag_without_prefix = file_tag.replace("Mathlib_", "")
    for kw, domain in DOMAIN_MAP:
        if kw in tag_without_prefix:
            return domain
    # Fall back to theorem name
    for kw, domain in DOMAIN_MAP:
        if kw in name:
            return domain
    return "other"


def make_theorem_id(name: str, idx: int) -> str:
    """Make a safe theorem_id from the name."""
    safe = name.replace(".", "_").replace(" ", "_")[:50]
    return f"ntp_{idx:04d}_{safe}"


def generate_canonical_nl(name: str, statement: str, client) -> str:
    """Generate a natural language YES/NO question from the Lean4 statement."""
    prompt = f"""Convert this Lean4 theorem to a clear natural language statement.

Theorem name: {name}
Lean4 statement: {statement}

Write a concise mathematical statement (1-2 sentences) that:
1. States what the theorem claims, without framing it as a question
2. Is understandable to a math PhD student
3. Uses standard mathematical English (not LaTeX)
4. Does NOT end with a question mark

Return ONLY the statement, nothing else."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/generated/ntp_mathlib_candidates.jsonl")
    parser.add_argument("--out", default="data/generated/ntp_mathlib_theorem_v1.jsonl")
    parser.add_argument(
        "--n",
        type=int,
        default=30,
        help="Number of theorems to convert (default: 30 for smoke test)",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index into candidates (for resuming)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Aborting.")
        sys.exit(1)

    import openai

    client = openai.OpenAI(api_key=api_key)

    candidates = [json.loads(l) for l in open(args.input)]
    slice_ = candidates[args.start : args.start + args.n]
    print(f"Converting {len(slice_)} candidates (indices {args.start}-{args.start + len(slice_) - 1})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing output to support resuming
    existing = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                existing[t["theorem_id"]] = t
        print(f"Resuming: {len(existing)} already converted")

    theorems = list(existing.values())
    errors = 0
    for i, cand in enumerate(slice_):
        global_idx = args.start + i
        theorem_id = make_theorem_id(cand["mathlib_name"], global_idx)

        if theorem_id in existing:
            print(f"  [{i + 1}/{len(slice_)}] SKIP (cached): {theorem_id}")
            continue

        print(f"  [{i + 1}/{len(slice_)}] {cand['mathlib_name'][:60]}")
        try:
            nl = generate_canonical_nl(cand["mathlib_name"], cand["lean4_statement"], client)
            domain = assign_domain(cand.get("file_tag", ""), cand["mathlib_name"])
            theorem = {
                "theorem_id": theorem_id,
                "mathlib_name": cand["mathlib_name"],
                "lean4_statement": cand["lean4_statement"],
                "canonical_nl": nl,
                "domain": domain,
                "tier": "T1",
                "cas_verifiable": False,
                "ground_truth": True,
                "seed": 42,
                "source": "ntp-mathlib",
                "file_tag": cand.get("file_tag", ""),
                "score": cand.get("score", 1.0),
            }
            theorems.append(theorem)
            print(f"         -> {nl[:80]}")
        except Exception as e:
            print(f"         ERROR: {e}")
            errors += 1

    # Write (overwrite to reflect latest set)
    with open(out_path, "w") as f:
        for t in theorems:
            f.write(json.dumps(t) + "\n")

    print(f"\nDone: {len(theorems)} theorems saved to {out_path}")
    print(f"Errors: {errors}")

    # Domain summary
    from collections import Counter

    domain_counts = Counter(t["domain"] for t in theorems)
    print("\nDomain distribution:")
    for domain, cnt in domain_counts.most_common():
        print(f"  {domain:<25} {cnt}")


if __name__ == "__main__":
    main()
