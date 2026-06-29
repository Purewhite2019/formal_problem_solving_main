import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the product of the $y$-coordinates of all the distinct solutions $(x,y)$ for the two equations $y=x^2-8$ and $y^2=-5x+44$.

## Answer
1736
-/

-- # Formal Answer
noncomputable abbrev Q161.gt := fun (answer : ℝ) => answer = (1736 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q161
find (answer : ℝ) s.t.
  (S : Finset (ℝ × ℝ))
  (h_sol : ∀ (p : ℝ × ℝ), p ∈ S ↔ p.2 = p.1 ^ 2 - 8 ∧ p.2 ^ 2 = -5 * p.1 + 44)
  (h_nonempty : ∃ (x y : ℝ), y = x ^ 2 - 8 ∧ y ^ 2 = -5 * x + 44)
: (answer = ∏ p in S, p.2)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q161
find_all (answer : ℝ) iff
  (S : Finset (ℝ × ℝ))
  (h_sol : ∀ (p : ℝ × ℝ), p ∈ S ↔ p.2 = p.1 ^ 2 - 8 ∧ p.2 ^ 2 = -5 * p.1 + 44)
  (h_nonempty : ∃ (x y : ℝ), y = x ^ 2 - 8 ∧ y ^ 2 = -5 * x + 44)
: (answer = ∏ p in S, p.2)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
