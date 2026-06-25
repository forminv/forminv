import Init.Data.Nat.Basic
import Init.Data.Nat.Div

-- F3: zero does not divide nonzero numbers
theorem f3_zero_not_divides_one : ¬ (0 ∣ (1 : Nat)) := by
  intro ⟨k, hk⟩
  simp at hk

-- F6: ge_iff_le for Nat
theorem f6_ge_iff_le (a b : Nat) : a ≥ b ↔ b ≤ a := Iff.rfl

-- n divides 0
theorem f3_n_divides_zero (n : Nat) : n ∣ 0 := ⟨0, rfl⟩

-- Simple disproof: 2 ∤ 1
theorem two_not_dvd_one : ¬ (2 ∣ (1 : Nat)) := by
  intro ⟨k, hk⟩
  omega

-- F5 disproof
theorem f5_biconditional_false :
    ¬ (∀ a m n : Nat, a ∣ (m + n) ↔ (a ∣ m ∧ a ∣ n)) := by
  intro h
  have key := h 2 1 1
  have hdvd : (2 : Nat) ∣ (1 + 1) := ⟨1, rfl⟩
  have hbad := key.mp hdvd
  exact two_not_dvd_one hbad.1

#check Nat.dvd_zero
#eval (2 : Nat) ∣ 2   -- should give true

