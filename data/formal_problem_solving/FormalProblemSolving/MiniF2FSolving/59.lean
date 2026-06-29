import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve for $a$: $\frac15|9+2a|<1$.  Express your answer in interval notation.

## Answer
(-7, -2)
-/

-- # Formal Answer
noncomputable abbrev Q59.gt := fun (answer : Set ℝ) => answer = (Set.Ioo (-7 : ℝ) (-2 : ℝ) : Set ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q59
find (answer : Set ℝ) s.t.
  (a : ℝ)
: (answer = {a | 1 / 5 * abs (9 + 2 * a) < 1})
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q59
find_all (answer : Set ℝ) iff
  (a : ℝ)
: (answer = {a | 1 / 5 * abs (9 + 2 * a) < 1})
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
