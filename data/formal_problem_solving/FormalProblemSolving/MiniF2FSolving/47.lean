import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the largest number $c$ such that $2x^2+5x+c=0$ has at least one real solution? Express your answer as a common fraction.

## Answer
$\frac{25}{8}$
-/

-- # Formal Answer
noncomputable abbrev Q47.gt := fun (answer : ℝ) => answer = (25 / 8 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q47
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {c : ℝ | ∃ x : ℝ, 2 * x^2 + 5 * x + c = 0})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q47
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {c : ℝ | ∃ x : ℝ, 2 * x^2 + 5 * x + c = 0})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
