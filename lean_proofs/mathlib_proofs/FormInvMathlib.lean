/-
  FormInv Paraphrase Equivalence Certificates -- Mathlib4 Edition
  AI4Math @ ICML 2026 | Verified: Lean 4.29.0 + Mathlib v4.29.0

  F7 CERTIFICATE: "prime" ↔ "has exactly two positive divisors"
    → Nat.prime_def: Prime p ↔ 2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

  F8 CERTIFICATE: addition commutativity / biconditional paraphrase
    → add_comm, Eq.symm

  F6 CERTIFICATE: Real.sqrt x ≥ 0 ↔ 0 ≤ Real.sqrt x
    → Real.sqrt_nonneg

  Companion to FormInvCertificates.lean (no-Mathlib proofs).
-/

import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

-- =============================================================
-- F7 CERTIFICATE -- "prime" ↔ "exactly two positive divisors"
-- =============================================================

/-- F7 CERTIFICATE: The Mathlib definition of Nat.Prime unpacks to
    exactly the "two-divisor" characterisation:
    p is prime ↔ p ≥ 2 ∧ every divisor of p is either 1 or p itself.

    This is Nat.prime_def in Mathlib verbatim. -/
theorem f7_prime_iff_two_divisors (p : ℕ) :
    p.Prime ↔ 2 ≤ p ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p :=
  Nat.prime_def

/-- F7 CERTIFICATE (split form): forward direction only --
    a prime has at most two divisors. -/
theorem f7_prime_divisors_forward (p : ℕ) (hp : p.Prime) (m : ℕ) (hm : m ∣ p) :
    m = 1 ∨ m = p :=
  hp.eq_one_or_self_of_dvd m hm

/-- F7 CERTIFICATE (split form): backward direction --
    two-le + two-divisors implies prime. -/
theorem f7_prime_divisors_backward (p : ℕ)
    (h1 : 2 ≤ p) (h2 : ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p) :
    p.Prime :=
  Nat.prime_def.mpr ⟨h1, h2⟩

-- =============================================================
-- F8 CERTIFICATE -- biconditional paraphrase of commutativity
-- =============================================================

/-- F8 CERTIFICATE (biconditional elimination):
    "a + b = b + a" and "b + a = a + b" are logically equivalent
    by symmetry of equality.

    This captures T1 paraphrase family F8 (domain-equivalent
    restatement by flipping both sides of an equation). -/
theorem f8_add_comm_bicon {α : Type*} [AddCommMonoid α] (a b : α) :
    a + b = b + a ↔ b + a = a + b :=
  ⟨fun h => h.symm, fun h => h.symm⟩

/-- F8 CERTIFICATE (instance):
    Addition is commutative -- the Mathlib ground truth that
    F8 paraphrases restate in alternative registers. -/
theorem f8_add_comm_cert {α : Type*} [AddCommMonoid α] (a b : α) :
    a + b = b + a :=
  add_comm a b

/-- F8 CERTIFICATE (Nat instance):
    Nat-specific commutativity for the benchmark items. -/
theorem f8_nat_add_comm (a b : ℕ) : a + b = b + a := Nat.add_comm a b

-- =============================================================
-- F6 CERTIFICATE -- Real.sqrt is nonneg (real analysis)
-- =============================================================

/-- F6 CERTIFICATE: "sqrt(x) ≥ 0" ↔ "0 ≤ sqrt(x)"
    Definitionally interchangeable (≥ is flip ≤).

    This is the T1 comparison-order paraphrase for the
    Real.sqrt_nonneg benchmark item. -/
theorem f6_sqrt_nonneg_iff (x : ℝ) :
    Real.sqrt x ≥ 0 ↔ 0 ≤ Real.sqrt x :=
  ge_iff_le

/-- F6 CERTIFICATE (ground truth):
    Real.sqrt x is always nonneg. -/
theorem f6_sqrt_nonneg (x : ℝ) : 0 ≤ Real.sqrt x :=
  Real.sqrt_nonneg x

-- =============================================================
-- SUMMARY TABLE
-- =============================================================
/-
  Proof                       | Family | Type       | Mathlib lemma used
  ----------------------------|--------|------------|---------------------------
  f7_prime_iff_two_divisors   |  F7    | T1 CERT    | Nat.prime_def
  f7_prime_divisors_forward   |  F7    | T1 CERT    | Prime.eq_one_or_self_of_dvd
  f7_prime_divisors_backward  |  F7    | T1 CERT    | Nat.prime_def.mpr
  f8_add_comm_bicon           |  F8    | T1 CERT    | Eq.symm (propositional)
  f8_add_comm_cert            |  F8    | T1 CERT    | add_comm
  f8_nat_add_comm             |  F8    | T1 CERT    | Nat.add_comm
  f6_sqrt_nonneg_iff          |  F6    | T1 CERT    | ge_iff_le
  f6_sqrt_nonneg              |  F6    | T1 CERT    | Real.sqrt_nonneg
-/
