import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the minimum possible value for $y$ in the equation $y = x^2 - 6x + 13$?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q132.gt := fun (answer : ℝ) => answer = (4 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q132
find (answer : ℝ) s.t.
: (∀ x : ℝ, answer ≤ x ^ 2 - 6 * x + 13) ∧ (∃ x : ℝ, x ^ 2 - 6 * x + 13 = answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q132
find_all (answer : ℝ) iff
: (∀ x : ℝ, answer ≤ x ^ 2 - 6 * x + 13) ∧ (∃ x : ℝ, x ^ 2 - 6 * x + 13 = answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
