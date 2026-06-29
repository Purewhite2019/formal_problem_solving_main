import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $n$ and $k$ are positive integers such that $5<\frac nk<6$, then what is the smallest possible value of $\frac{\mathop{\text{lcm}}[n,k]}{\gcd(n,k)}$?

## Answer
22
-/

-- # Formal Answer
noncomputable abbrev Q136.gt := fun (answer : ℕ) => answer = (22 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q136
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (n : ℕ)
  (k : ℕ)
  (h₀ : 0 < n ∧ 0 < k)
  (h₁ : (5 : ℝ) < (n : ℝ) / k)
  (h₂ : (n : ℝ) / k < 6)
  (hS : S = {m | ∃ n k : ℕ, 0 < n ∧ 0 < k ∧ 5 < (n : ℝ) / k ∧ (n : ℝ) / k < 6 ∧ m = (Nat.lcm n k) / (Nat.gcd n k)})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q136
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (n : ℕ)
  (k : ℕ)
  (h₀ : 0 < n ∧ 0 < k)
  (h₁ : (5 : ℝ) < (n : ℝ) / k)
  (h₂ : (n : ℝ) / k < 6)
  (hS : S = {m | ∃ n k : ℕ, 0 < n ∧ 0 < k ∧ 5 < (n : ℝ) / k ∧ (n : ℝ) / k < 6 ∧ m = (Nat.lcm n k) / (Nat.gcd n k)})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
