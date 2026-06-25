-- Quick test of core Lean4 proofs (no Mathlib)
theorem f5_err1_dvd_add_biconditional_is_false :
    ¬ (∀ a m n : ℕ, a ∣ (m + n) ↔ (a ∣ m ∧ a ∣ n)) := by
  intro h
  have key : 2 ∣ (1 + 1) ↔ (2 ∣ 1 ∧ 2 ∣ 1) := h 2 1 1
  have hdvd : 2 ∣ (1 + 1) := by norm_num
  have hbad := key.mp hdvd
  have : 2 ∣ 1 := hbad.1
  omega

theorem f3_zero_not_divides_one : ¬ (0 ∣ 1) := by decide

theorem f6_ge_iff_le_nat (a b : ℕ) : a ≥ b ↔ b ≤ a := Iff.rfl

theorem f6_prime_two_le (p : ℕ) : p ≥ 2 ↔ 2 ≤ p := Iff.rfl

theorem f3_n_divides_zero (n : ℕ) : n ∣ 0 := dvd_zero n

#check @Nat.mod_self

-- Quick sanity: is the biconditional actually false?
#eval (2 % (1 + 1))   -- should be 0 (2 divides 2)
#eval (1 % 2)          -- should be 1 (2 does not divide 1)

