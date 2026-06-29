import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the number of integer values of $k$ in the closed interval $[-500,500]$ for which the equation $\log(kx)=2\log(x+2)$ has exactly one real solution.

## Answer
501
-/

-- # Formal Answer
noncomputable abbrev Q340.gt := fun (answer : ℕ) => answer = (501 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q340
find (answer : ℕ) s.t.
  (S : Set ℤ)
  (hS : S = {k:ℤ | k ≤ 500 ∧ k ≥ -500 ∧ (∃ x : ℝ, k*x > 0 ∧ x + 2 > 0  ∧ Real.log (k * x) = 2 * Real.log (x + 2)
  ∧ (∀ a : ℝ, (k*a > 0 ∧ a + 2 > 0  ∧ Real.log (k * a) = 2 * Real.log (a + 2)) → a = x))})
: (answer  = S.encard)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q340
find_all (answer : ℕ) iff
  (S : Set ℤ)
  (hS : S = {k:ℤ | k ≤ 500 ∧ k ≥ -500 ∧ (∃ x : ℝ, k*x > 0 ∧ x + 2 > 0  ∧ Real.log (k * x) = 2 * Real.log (x + 2)
  ∧ (∀ a : ℝ, (k*a > 0 ∧ a + 2 > 0  ∧ Real.log (k * a) = 2 * Real.log (a + 2)) → a = x))})
: (answer  = S.encard)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
