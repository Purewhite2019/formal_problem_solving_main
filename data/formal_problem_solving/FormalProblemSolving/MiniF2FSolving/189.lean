import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the positive difference between $120\%$ of 30 and $130\%$ of 20?

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q189.gt := fun (answer : ℝ) => answer = (10 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q189
find (answer : ℝ) s.t.
: (answer = abs ((120 : ℝ)/100 * 30 - 130/100 * 20))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q189
find_all (answer : ℝ) iff
: (answer = abs ((120 : ℝ)/100 * 30 - 130/100 * 20))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
