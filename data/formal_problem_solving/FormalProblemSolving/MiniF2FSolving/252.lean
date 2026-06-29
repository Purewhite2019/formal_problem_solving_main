import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the radius of the circle with equation $x^2 + 8x + y^2 - 6y = 0$.

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q252.gt := fun (answer : ℝ) => answer = (5 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q252
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : x^2 + 8 * x + y^2 - 6 * y = 0)
: (answer ≥ 0) ∧ ((x + 4)^2 + (y-3)^2 = answer^2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q252
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : x^2 + 8 * x + y^2 - 6 * y = 0)
: (answer ≥ 0) ∧ ((x + 4)^2 + (y-3)^2 = answer^2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
