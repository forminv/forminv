"""Mathlib theorem loader and curated theorem set for gate check.

Holds :data:`CURATED_THEOREMS`, a hand-crafted set of real Mathlib theorems
(with canonical natural-language phrasings spanning Tiers 1-3), and loaders that
turn them into :class:`~forminv.schemas.BaseTheorem` objects or pull additional
raw theorem rows from HuggingFace for scaling up.
"""

from forminv.schemas import BaseTheorem, DomainTier

# -- CURATED GATE-CHECK SET (50 theorems, no Lean4 needed) ------------------
# These are real Mathlib theorems with hand-crafted canonical NL phrasings.
# Covers Tier 1-2. All ground_truth=True (valid theorems).
# Format: (mathlib_name, lean4_statement, canonical_nl, domain, tier, cas_verifiable)

CURATED_THEOREMS = [
    # -- Number Theory T1 --------------------------------------------------
    (
        "Nat.Prime.two_le",
        "theorem Nat.Prime.two_le : ∀ {p : ℕ}, p.Prime → 2 ≤ p",
        "Every prime number is at least 2.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.Prime.one_lt",
        "theorem Nat.Prime.one_lt : ∀ {p : ℕ}, p.Prime → 1 < p",
        "Every prime number is greater than 1.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.not_prime_one",
        "theorem Nat.not_prime_one : ¬Nat.Prime 1",
        "The number 1 is not prime.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.Prime.prime_two",
        "theorem Nat.prime_def : Nat.Prime 2",
        "2 is a prime number.",
        "number_theory",
        DomainTier.T1,
        True,
    ),
    (
        "Nat.dvd_antisymm",
        "theorem Nat.dvd_antisymm : ∀ {m n : ℕ}, m ∣ n → n ∣ m → m = n",
        "If two natural numbers each divide the other, they are equal.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.dvd_refl",
        "theorem Nat.dvd_refl : ∀ (n : ℕ), n ∣ n",
        "Every natural number divides itself.",
        "number_theory",
        DomainTier.T1,
        True,
    ),
    (
        "Nat.dvd_zero",
        "theorem Nat.dvd_zero : ∀ (n : ℕ), n ∣ 0",
        "Every natural number divides zero.",
        "number_theory",
        DomainTier.T1,
        True,
    ),
    (
        "Nat.zero_dvd",
        "theorem Nat.zero_dvd : ∀ {n : ℕ}, 0 ∣ n ↔ n = 0",
        "Zero divides a natural number if and only if that number is zero.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.dvd_add",
        "theorem Nat.dvd_add : ∀ {k m n : ℕ}, k ∣ m → k ∣ n → k ∣ m + n",
        "If a number divides two numbers, it divides their sum.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.Prime.dvd_mul",
        "theorem Nat.Prime.dvd_mul : ∀ {p m n : ℕ}, p.Prime → p ∣ m * n → p ∣ m ∨ p ∣ n",
        "If a prime divides a product, it divides at least one factor.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    # -- Modular Arithmetic T1 --------------------------------------------
    (
        "Int.emod_emod_of_dvd",
        "theorem Int.emod_add_ediv : ∀ (a b : ℤ), a % b + b * (a / b) = a",
        "For integers a and b, we have a mod b + b * (a div b) = a.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.mod_self",
        "theorem Nat.mod_self : ∀ (n : ℕ), n % n = 0",
        "Any natural number modulo itself is zero.",
        "number_theory",
        DomainTier.T1,
        True,
    ),
    (
        "Nat.zero_mod",
        "theorem Nat.zero_mod : ∀ (b : ℕ), 0 % b = 0",
        "Zero modulo any natural number is zero.",
        "number_theory",
        DomainTier.T1,
        True,
    ),
    # -- Basic Algebra T1 ------------------------------------------------
    (
        "sq_nonneg",
        "theorem sq_nonneg : ∀ {α : Type u_1} [inst : LinearOrderedSemiring α] (a : α), 0 ≤ a ^ 2",
        "The square of any real number is nonnegative.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "add_comm",
        "theorem add_comm : ∀ {G : Type u_1} [inst : AddCommMonoid G] (a b : G), a + b = b + a",
        "Addition is commutative.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "mul_comm",
        "theorem mul_comm : ∀ {G : Type u_1} [inst : CommMonoid G] (a b : G), a * b = b * a",
        "Multiplication is commutative.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "add_assoc",
        "theorem add_assoc : ∀ {G : Type u_1} [inst : AddSemigroup G] (a b c : G), a + b + c = a + (b + c)",
        "Addition is associative.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "mul_assoc",
        "theorem mul_assoc : ∀ {G : Type u_1} [inst : Semigroup G] (a b c : G), a * b * c = a * (b * c)",
        "Multiplication is associative.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "add_zero",
        "theorem add_zero : ∀ {M : Type u_1} [inst : AddZeroClass M] (a : M), a + 0 = a",
        "Adding zero to any element returns that element.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "zero_add",
        "theorem zero_add : ∀ {M : Type u_1} [inst : AddZeroClass M] (a : M), 0 + a = a",
        "Zero plus any element is that element.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "mul_one",
        "theorem mul_one : ∀ {M : Type u_1} [inst : MulOneClass M] (a : M), a * 1 = a",
        "Any element multiplied by one is itself.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "one_mul",
        "theorem one_mul : ∀ {M : Type u_1} [inst : MulOneClass M] (a : M), 1 * a = a",
        "One times any element is that element.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "sq_abs",
        "theorem sq_abs : ∀ {α : Type u_1} [inst : LinearOrderedRing α] (a : α), |a| ^ 2 = a ^ 2",
        "The square of the absolute value equals the square.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "abs_nonneg",
        "theorem abs_nonneg : ∀ {α : Type u_1} [inst : LinearOrderedAddCommGroup α] (a : α), 0 ≤ |a|",
        "The absolute value of any element is nonnegative.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    # -- Polynomial Algebra T2 --------------------------------------------
    (
        "Polynomial.degree_add_le",
        "theorem Polynomial.degree_add_le : ∀ {R : Type u_1} [inst : Semiring R] (p q : R[X]), (p + q).degree ≤ max p.degree q.degree",
        "The degree of a sum of polynomials is at most the maximum of their degrees.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    (
        "Polynomial.degree_mul",
        "theorem Polynomial.degree_mul : ∀ {R : Type u_1} [inst : CommSemiring R] [inst_1 : NoZeroDivisors R] (p q : R[X]), (p * q).degree = p.degree + q.degree",
        "The degree of a product of polynomials equals the sum of their degrees (over an integral domain).",
        "algebra",
        DomainTier.T2,
        False,
    ),
    # -- Set Theory T2 ----------------------------------------------------
    (
        "Set.subset_refl",
        "theorem Set.subset_refl : ∀ {α : Type u_1} (s : Set α), s ⊆ s",
        "Every set is a subset of itself.",
        "set_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Set.subset_trans",
        "theorem Set.subset_trans : ∀ {α : Type u_1} {s t u : Set α}, s ⊆ t → t ⊆ u → s ⊆ u",
        "Subset is transitive.",
        "set_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Set.inter_comm",
        "theorem Set.inter_comm : ∀ {α : Type u_1} (s t : Set α), s ∩ t = t ∩ s",
        "Set intersection is commutative.",
        "set_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Set.union_comm",
        "theorem Set.union_comm : ∀ {α : Type u_1} (s t : Set α), s ∪ t = t ∪ s",
        "Set union is commutative.",
        "set_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Set.inter_assoc",
        "theorem Set.inter_assoc : ∀ {α : Type u_1} (s t u : Set α), s ∩ t ∩ u = s ∩ (t ∩ u)",
        "Set intersection is associative.",
        "set_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Set.union_assoc",
        "theorem Set.union_assoc : ∀ {α : Type u_1} (s t u : Set α), s ∪ t ∪ u = s ∪ (t ∪ u)",
        "Set union is associative.",
        "set_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Set.empty_subset",
        "theorem Set.empty_subset : ∀ {α : Type u_1} (s : Set α), ∅ ⊆ s",
        "The empty set is a subset of every set.",
        "set_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Set.inter_empty",
        "theorem Set.inter_empty : ∀ {α : Type u_1} (s : Set α), s ∩ ∅ = ∅",
        "The intersection of any set with the empty set is empty.",
        "set_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Set.union_empty",
        "theorem Set.union_empty : ∀ {α : Type u_1} (s : Set α), s ∪ ∅ = s",
        "The union of any set with the empty set is that set.",
        "set_theory",
        DomainTier.T1,
        False,
    ),
    # -- Logic / Finset T2 ------------------------------------------------
    (
        "Finset.card_empty",
        "theorem Finset.card_empty : Finset.card (∅ : Finset α) = 0",
        "The cardinality of the empty finite set is zero.",
        "combinatorics",
        DomainTier.T1,
        True,
    ),
    (
        "Finset.card_singleton",
        "theorem Finset.card_singleton : ∀ {α : Type u_1} (a : α), ({a} : Finset α).card = 1",
        "A singleton finite set has cardinality 1.",
        "combinatorics",
        DomainTier.T1,
        True,
    ),
    (
        "Nat.factorial_pos",
        "theorem Nat.factorial_pos : ∀ (n : ℕ), 0 < n !",
        "The factorial of any natural number is positive.",
        "combinatorics",
        DomainTier.T1,
        True,
    ),
    (
        "Nat.choose_symm",
        "theorem Nat.choose_symm : ∀ {n k : ℕ}, k ≤ n → Nat.choose n (n - k) = Nat.choose n k",
        "Binomial coefficients are symmetric: C(n, k) = C(n, n-k).",
        "combinatorics",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.choose_zero_right",
        "theorem Nat.choose_zero_right : ∀ (n : ℕ), Nat.choose n 0 = 1",
        "C(n, 0) = 1 for all natural numbers n.",
        "combinatorics",
        DomainTier.T1,
        True,
    ),
    (
        "Nat.choose_self",
        "theorem Nat.choose_self : ∀ (n : ℕ), Nat.choose n n = 1",
        "C(n, n) = 1 for all natural numbers n.",
        "combinatorics",
        DomainTier.T1,
        True,
    ),
    # -- Real Analysis T2 -------------------------------------------------
    (
        "Real.sqrt_nonneg",
        "theorem Real.sqrt_nonneg : ∀ (x : ℝ), 0 ≤ Real.sqrt x",
        "The real square root function is always nonnegative.",
        "analysis",
        DomainTier.T2,
        True,
    ),
    (
        "Real.sqrt_sq",
        "theorem Real.sqrt_sq : ∀ {x : ℝ}, 0 ≤ x → Real.sqrt (x ^ 2) = x",
        "For nonneg x, the square root of x squared is x.",
        "analysis",
        DomainTier.T2,
        False,
    ),
    (
        "Real.sqrt_mul_self",
        "theorem Real.sqrt_mul_self : ∀ {x : ℝ}, 0 ≤ x → Real.sqrt x * Real.sqrt x = x",
        "For nonneg x, sqrt(x) * sqrt(x) = x.",
        "analysis",
        DomainTier.T2,
        False,
    ),
    # -- Group Theory T2 --------------------------------------------------
    (
        "mul_inv_cancel",
        "theorem mul_inv_cancel : ∀ {G : Type u_1} [inst : Group G] (a : G), a * a^-1 = 1",
        "In a group, every element times its inverse is the identity.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    (
        "inv_mul_cancel",
        "theorem inv_mul_cancel : ∀ {G : Type u_1} [inst : Group G] (a : G), a^-1 * a = 1",
        "In a group, the inverse of an element times the element is the identity.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    (
        "inv_inv",
        "theorem inv_inv : ∀ {G : Type u_1} [inst : Group G] (a : G), a^-1^-1 = a",
        "The inverse of the inverse of an element is the element itself.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    (
        "mul_left_cancel",
        "theorem mul_left_cancel : ∀ {G : Type u_1} [inst : LeftCancelMonoid G] {a b c : G}, a * b = a * c → b = c",
        "In a group, if a*b = a*c then b = c (left cancellation).",
        "algebra",
        DomainTier.T2,
        False,
    ),
    # -- Order Theory T2 --------------------------------------------------
    (
        "le_refl",
        "theorem le_refl : ∀ {α : Type u_1} [inst : Preorder α] (a : α), a ≤ a",
        "Every element is less than or equal to itself.",
        "order_theory",
        DomainTier.T1,
        False,
    ),
    (
        "le_trans",
        "theorem le_trans : ∀ {α : Type u_1} [inst : Preorder α] {a b c : α}, a ≤ b → b ≤ c → a ≤ c",
        "The ≤ relation is transitive.",
        "order_theory",
        DomainTier.T1,
        False,
    ),
    (
        "le_antisymm",
        "theorem le_antisymm : ∀ {α : Type u_1} [inst : PartialOrder α] {a b : α}, a ≤ b → b ≤ a → a = b",
        "In a partial order, if a ≤ b and b ≤ a then a = b.",
        "order_theory",
        DomainTier.T1,
        False,
    ),
    (
        "lt_irrefl",
        "theorem lt_irrefl : ∀ {α : Type u_1} [inst : Preorder α] (a : α), ¬a < a",
        "No element is strictly less than itself.",
        "order_theory",
        DomainTier.T1,
        False,
    ),
    # -- EXPANSION: Targeting 6 known failure modes ------------------------
    # Mode F3 (active/passive): relational theorems with direction
    (
        "Nat.pos_of_ne_zero",
        "theorem Nat.pos_of_ne_zero : ∀ {n : ℕ}, n ≠ 0 → 0 < n",
        "A nonzero natural number is positive.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.lt_of_lt_of_le",
        "theorem Nat.lt_of_lt_of_le : ∀ {a b c : ℕ}, a < b → b ≤ c → a < c",
        "If a < b and b ≤ c, then a < c.",
        "order_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.le_of_lt",
        "theorem Nat.le_of_lt : ∀ {a b : ℕ}, a < b → a ≤ b",
        "Strict inequality implies weak inequality.",
        "order_theory",
        DomainTier.T1,
        False,
    ),
    # Mode F4 (formal notation): theorems with symbolic forms
    (
        "Int.neg_neg",
        "theorem Int.neg_neg : ∀ (a : ℤ), -(-a) = a",
        "The negation of a negation is the original integer.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "Int.add_neg_cancel",
        "theorem Int.add_neg_cancel : ∀ (a : ℤ), a + (-a) = 0",
        "An integer plus its negation equals zero.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "Int.mul_neg_one",
        "theorem Int.mul_neg_one : ∀ (a : ℤ), a * (-1) = -a",
        "An integer times negative one equals its negation.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    # Mode F5 (connective variation): iff/biconditional theorems
    (
        "Nat.even_iff",
        "theorem Nat.even_iff : ∀ {n : ℕ}, Even n ↔ n % 2 = 0",
        "A natural number is even if and only if it is divisible by 2.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.odd_iff",
        "theorem Nat.odd_iff : ∀ {n : ℕ}, Odd n ↔ n % 2 = 1",
        "A natural number is odd if and only if its remainder when divided by 2 is 1.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Int.even_iff_not_odd",
        "theorem Int.even_iff_not_odd : ∀ (n : ℤ), Even n ↔ ¬Odd n",
        "An integer is even if and only if it is not odd.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    # Mode F6 (comparison order): inequalities that can be reversed
    (
        "abs_le",
        "theorem abs_le : ∀ {α : Type u_1} [inst : LinearOrderedAddCommGroup α] {a b : α}, |a| ≤ b ↔ -b ≤ a ∧ a ≤ b",
        "|a| ≤ b if and only if -b ≤ a and a ≤ b.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    (
        "neg_le_iff_add_nonneg",
        "theorem neg_le_iff_add_nonneg : ∀ {α : Type u_1} [inst : OrderedAddCommGroup α] {a b : α}, -a ≤ b ↔ 0 ≤ a + b",
        "-a ≤ b if and only if 0 ≤ a + b.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    (
        "Nat.lt_succ_iff",
        "theorem Nat.lt_succ_iff : ∀ {m n : ℕ}, m < n + 1 ↔ m ≤ n",
        "m < n+1 if and only if m ≤ n.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    # Mode F2 (quantifier variation): all/every/any/each
    (
        "Nat.succ_pos",
        "theorem Nat.succ_pos : ∀ (n : ℕ), 0 < n + 1",
        "The successor of every natural number is positive.",
        "number_theory",
        DomainTier.T1,
        True,
    ),
    (
        "Nat.zero_le",
        "theorem Nat.zero_le : ∀ (n : ℕ), 0 ≤ n",
        "Zero is less than or equal to every natural number.",
        "number_theory",
        DomainTier.T1,
        True,
    ),
    (
        "mul_pos",
        "theorem mul_pos : ∀ {α : Type u_1} [inst : StrictOrderedSemiring α] {a b : α}, 0 < a → 0 < b → 0 < a * b",
        "The product of two positive numbers is positive.",
        "algebra",
        DomainTier.T1,
        False,
    ),
    # Mode F7 (definitional unpack): theorems where definition matters
    (
        "Nat.Prime.eq_one_or_self_of_dvd",
        "theorem Nat.Prime.eq_one_or_self_of_dvd : ∀ {p : ℕ}, p.Prime → ∀ (a : ℕ), a ∣ p → a = 1 ∨ a = p",
        "The only divisors of a prime number are 1 and itself.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.Prime.ne_one",
        "theorem Nat.Prime.ne_one : ∀ {p : ℕ}, p.Prime → p ≠ 1",
        "No prime number equals 1.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.not_prime_zero",
        "theorem Nat.not_prime_zero : ¬Nat.Prime 0",
        "0 is not a prime number.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    # F8 (domain equivalent): same pattern in different domains
    (
        "List.length_nil",
        "theorem List.length_nil : List.length ([] : List α) = 0",
        "The empty list has length zero.",
        "combinatorics",
        DomainTier.T1,
        True,
    ),
    (
        "List.length_cons",
        "theorem List.length_cons : ∀ {α : Type u_1} (a : α) (l : List α), (a :: l).length = l.length + 1",
        "Appending an element to a list increases its length by 1.",
        "combinatorics",
        DomainTier.T1,
        False,
    ),
    (
        "List.append_nil",
        "theorem List.append_nil : ∀ {α : Type u_1} (l : List α), l ++ [] = l",
        "Appending the empty list to any list leaves it unchanged.",
        "combinatorics",
        DomainTier.T1,
        False,
    ),
    (
        "List.nil_append",
        "theorem List.nil_append : ∀ {α : Type u_1} (l : List α), [] ++ l = l",
        "Prepending the empty list to any list leaves it unchanged.",
        "combinatorics",
        DomainTier.T1,
        False,
    ),
    # -- More number theory (Tier 2) -------------------------------------
    (
        "Nat.gcd_comm",
        "theorem Nat.gcd_comm : ∀ (m n : ℕ), Nat.gcd m n = Nat.gcd n m",
        "The greatest common divisor is commutative.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.gcd_self",
        "theorem Nat.gcd_self : ∀ (n : ℕ), Nat.gcd n n = n",
        "The GCD of a number with itself is the number.",
        "number_theory",
        DomainTier.T1,
        True,
    ),
    (
        "Nat.gcd_zero_right",
        "theorem Nat.gcd_zero_right : ∀ (n : ℕ), Nat.gcd n 0 = n",
        "The GCD of any number with 0 is the number itself.",
        "number_theory",
        DomainTier.T1,
        True,
    ),
    (
        "Nat.lcm_comm",
        "theorem Nat.lcm_comm : ∀ (m n : ℕ), Nat.lcm m n = Nat.lcm n m",
        "The least common multiple is commutative.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    (
        "Nat.dvd_lcm_left",
        "theorem Nat.dvd_lcm_left : ∀ (m n : ℕ), m ∣ Nat.lcm m n",
        "Every number divides the LCM of itself and any other number.",
        "number_theory",
        DomainTier.T1,
        False,
    ),
    # -- More algebra T2 --------------------------------------------------
    (
        "neg_add_cancel",
        "theorem neg_add_cancel : ∀ {G : Type u_1} [inst : AddGroup G] (a : G), -a + a = 0",
        "The negative of an element plus the element equals zero.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    (
        "sub_self",
        "theorem sub_self : ∀ {G : Type u_1} [inst : AddGroup G] (a : G), a - a = 0",
        "Any element minus itself equals zero.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "mul_zero",
        "theorem mul_zero : ∀ {M_0 : Type u_1} [inst : MulZeroClass M_0] (a : M_0), a * 0 = 0",
        "Any element times zero equals zero.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "zero_mul",
        "theorem zero_mul : ∀ {M_0 : Type u_1} [inst : MulZeroClass M_0] (a : M_0), 0 * a = 0",
        "Zero times any element equals zero.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "neg_zero",
        "theorem neg_zero : -(0 : G) = 0",
        "The negation of zero is zero.",
        "algebra",
        DomainTier.T1,
        True,
    ),
    (
        "mul_neg",
        "theorem mul_neg : ∀ {α : Type u_1} [inst : NonUnitalNonAssocRing α] (a b : α), a * (-b) = -(a * b)",
        "a times negative b equals the negative of a times b.",
        "algebra",
        DomainTier.T1,
        False,
    ),
    (
        "neg_mul",
        "theorem neg_mul : ∀ {α : Type u_1} [inst : NonUnitalNonAssocRing α] (a b : α), (-a) * b = -(a * b)",
        "Negative a times b equals the negative of a times b.",
        "algebra",
        DomainTier.T1,
        False,
    ),
    # -- More analysis T2 -------------------------------------------------
    (
        "Real.abs_abs",
        "theorem Real.abs_abs : ∀ (a : ℝ), ||(a)|| = |a|",
        "The absolute value of the absolute value is the absolute value.",
        "analysis",
        DomainTier.T2,
        False,
    ),
    (
        "Real.abs_zero",
        "theorem abs_zero : |0| = 0",
        "The absolute value of zero is zero.",
        "analysis",
        DomainTier.T1,
        True,
    ),
    (
        "Real.abs_neg",
        "theorem abs_neg : ∀ {α : Type u_1} [inst : LinearOrderedAddCommGroup α] (a : α), |(-a)| = |a|",
        "The absolute value of a negative equals the absolute value of the positive.",
        "analysis",
        DomainTier.T1,
        False,
    ),
    (
        "Real.abs_mul",
        "theorem abs_mul : ∀ {α : Type u_1} [inst : LinearOrderedRing α] (a b : α), |a * b| = |a| * |b|",
        "The absolute value of a product equals the product of the absolute values.",
        "analysis",
        DomainTier.T1,
        False,
    ),
    (
        "Real.triangle_inequality",
        "theorem abs_add : ∀ {α : Type u_1} [inst : LinearOrderedAddCommGroup α] (a b : α), |a + b| ≤ |a| + |b|",
        "The triangle inequality: |a + b| ≤ |a| + |b|.",
        "analysis",
        DomainTier.T1,
        False,
    ),
    # -- Topology T3 (harder) ---------------------------------------------
    (
        "IsOpen.union",
        "theorem IsOpen.union : ∀ {X : Type u_1} [inst : TopologicalSpace X] {s t : Set X}, IsOpen s → IsOpen t → IsOpen (s ∪ t)",
        "The union of two open sets is open.",
        "topology",
        DomainTier.T3,
        False,
    ),
    (
        "IsOpen.inter",
        "theorem IsOpen.inter : ∀ {X : Type u_1} [inst : TopologicalSpace X] {s t : Set X}, IsOpen s → IsOpen t → IsOpen (s ∩ t)",
        "The intersection of two open sets is open.",
        "topology",
        DomainTier.T3,
        False,
    ),
    (
        "isClosed_empty",
        "theorem isClosed_empty : IsClosed (∅ : Set X)",
        "The empty set is closed.",
        "topology",
        DomainTier.T3,
        False,
    ),
    (
        "isClosed_univ",
        "theorem isClosed_univ : IsClosed (Set.univ : Set X)",
        "The universal set is closed.",
        "topology",
        DomainTier.T3,
        False,
    ),
    (
        "isOpen_compl_iff",
        "theorem isOpen_compl_iff : ∀ {X : Type u_1} [inst : TopologicalSpace X] {s : Set X}, IsOpen s^c ↔ IsClosed s",
        "The complement of a set is open if and only if the set is closed.",
        "topology",
        DomainTier.T3,
        False,
    ),
    # -- Group Theory T2 -------------------------------------------------
    (
        "mul_left_inj",
        "theorem mul_left_inj : ∀ {G : Type u_1} [inst : Group G] {a b c : G}, a * b = a * c ↔ b = c",
        "Left multiplication is injective: a*b = a*c if and only if b = c.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    (
        "mul_right_inj",
        "theorem mul_right_inj : ∀ {G : Type u_1} [inst : Group G] {a b c : G}, b * a = c * a ↔ b = c",
        "Right multiplication is injective: b*a = c*a if and only if b = c.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    (
        "one_inv",
        "theorem one_inv : (1 : G)^-1 = 1",
        "The inverse of the identity is the identity.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    (
        "mul_inv_rev",
        "theorem mul_inv_rev : ∀ {G : Type u_1} [inst : Group G] (a b : G), (a * b)^-1 = b^-1 * a^-1",
        "The inverse of a product is the product of inverses in reverse order.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    # -- Field Theory T3 -------------------------------------------------
    (
        "div_self",
        "theorem div_self : ∀ {α : Type u_1} [inst : DivisionRing α] {a : α}, a ≠ 0 → a / a = 1",
        "Any nonzero element divided by itself equals 1.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    (
        "mul_div_cancel",
        "theorem mul_div_cancel_0 : ∀ {α : Type u_1} [inst : DivisionRing α] (a : α) {b : α}, b ≠ 0 → a * b / b = a",
        "a times b divided by b equals a, when b is nonzero.",
        "algebra",
        DomainTier.T2,
        False,
    ),
    (
        "inv_mul_cancel",
        "theorem inv_mul_cancel_0 : ∀ {α : Type u_1} [inst : GroupWithZero α] {a : α}, a ≠ 0 → a^-1 * a = 1",
        "The multiplicative inverse of a nonzero element times the element equals 1.",
        "algebra",
        DomainTier.T2,
        False,
    ),
]


def load_curated_theorems(
    tier_filter: list[DomainTier] | None = None,
    domain_filter: list[str] | None = None,
    limit: int | None = None,
) -> list[BaseTheorem]:
    """Load the curated gate-check theorem set as ``BaseTheorem`` objects.

    Args:
        tier_filter: If given, keep only theorems whose tier is in this list.
        domain_filter: If given, keep only theorems whose domain is in this list.
        limit: If given, stop after collecting this many theorems.

    Returns:
        The curated theorems (after filtering) as
        :class:`~forminv.schemas.BaseTheorem` instances with generated IDs.
    """
    theorems = []
    for i, (name, lean4, nl, domain, tier, cas) in enumerate(CURATED_THEOREMS):
        if tier_filter and tier not in tier_filter:
            continue
        if domain_filter and domain not in domain_filter:
            continue
        theorems.append(
            BaseTheorem(
                theorem_id=f"thm_{i:04d}_{name.replace('.', '_').lower()[:20]}",
                lean4_statement=lean4,
                canonical_nl=nl,
                ground_truth=True,
                domain=domain,
                tier=tier,
                mathlib_name=name,
                cas_verifiable=cas,
                seed=42,
            )
        )
        if limit and len(theorems) >= limit:
            break
    return theorems


def load_from_huggingface(
    dataset_name: str = "JohnYang88/lean-dojo-mathlib4", split: str = "train", limit: int = 200
) -> list[dict]:
    """Stream raw theorem rows from a HuggingFace dataset (for scaling up).

    Args:
        dataset_name: HuggingFace dataset identifier to stream from.
        split: Dataset split to read.
        limit: Maximum number of rows to return.

    Returns:
        Up to ``limit`` raw dataset rows as dictionaries.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)
    rows = []
    for row in ds:
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows
