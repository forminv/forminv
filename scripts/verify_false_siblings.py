#!/usr/bin/env python3
"""Machine-check the numeric counterexamples in false_siblings_50.json.

Every 'nat'-type check is verified deterministically so we never ship a FALSE
sibling that is accidentally true. 'manual' checks are listed for human/Lean
review (most already have Lean4 counterexamples in the supplementary proofs).
"""

import json
import math
import os

from sympy import binomial, isprime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIB = os.path.join(ROOT, "data/generated/false_siblings_50.json")


def check(c):
    t = c.get("expr")
    if t == "all_primes_ge":  # claim: every prime >= k ; witness prime < k
        w = c["witness"]
        return isprime(w) and w < c["k"]
    if t == "all_primes_gt":  # claim: every prime > k ; witness prime not > k
        w = c["witness"]
        return isprime(w) and not (w > c["k"])
    if t == "is_prime":  # claim n is prime; expect false
        return isprime(c["n"]) == c["expect"]
    if t == "divides":  # claim a | b ; expect false means a does NOT divide b
        a, b = c["a"], c["b"]
        actually = (b % a == 0) if a != 0 else (b == 0)
        return actually == c["expect"]
    if t == "mod":  # claim a mod b == claim ; truth is expect_eq
        return (c["a"] % c["b"]) == c["expect_eq"] and c["expect_eq"] != c["claim"]
    if t == "factorial_gt_n":  # claim n! > n ; witness violates
        w = c["witness"]
        return not (math.factorial(w) > w)
    if t == "choose":
        if "swap_n" in c:  # claim C(n,k)=C(k,n)
            return binomial(c["n"], c["k"]) != binomial(c["swap_n"], c["swap_k"])
        return binomial(c["n"], c["k"]) == c["expect_eq"] and c["expect_eq"] != c["claim"]
    if t == "monoid_cancel":  # a*b == a*c but b != c
        return c["a"] * c["b"] == c["a"] * c["c"] and c["b"] != c["c"]
    if t == "le_symm":  # a<=b but not b<=a
        return c["a"] <= c["b"] and not (c["b"] <= c["a"])
    return None


def main():
    sibs = json.load(open(SIB))["siblings"]
    n_nat = n_pass = n_manual = 0
    fails = []
    for s in sibs:
        ch = s["check"]
        if ch["type"] == "nat":
            n_nat += 1
            ok = check(ch)
            if ok:
                n_pass += 1
            else:
                fails.append((s["theorem_id"], ok))
        else:
            n_manual += 1
    print(f"Total siblings        : {len(sibs)}")
    print(f"Numeric (machine-chk) : {n_nat}  passed: {n_pass}  failed: {len(fails)}")
    print(f"Manual/Lean review    : {n_manual}")
    if fails:
        print("\nFAILED (sibling may be accidentally TRUE -- fix before use):")
        for tid, r in fails:
            print(f"  {tid}: check returned {r}")
        raise SystemExit(1)
    print("\nAll numeric counterexamples verified. Manual ones listed for Lean/expert review.")


if __name__ == "__main__":
    main()
