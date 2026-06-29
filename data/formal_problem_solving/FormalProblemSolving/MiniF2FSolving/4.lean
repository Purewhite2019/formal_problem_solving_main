import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Expand the following expression: $7(3y+2)$

## Answer
21y + 14
-/

-- # Formal Answer
noncomputable abbrev Q4.gt := fun (y : ℂ) (answer : ℂ) => answer = (((21 : ℂ) * y : ℂ) + (14 : ℂ) : ℂ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q4
  (y : ℂ)
find (answer : ℂ) s.t.
: (answer = 7 * (3 * y + 2))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q4
  (y : ℂ)
find_all (answer : ℂ) iff
: (answer = 7 * (3 * y + 2))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
