import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Evaluate $\log_327$.

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q297.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q297
find (answer : ℝ) s.t.
: (answer = Real.log 27 / Real.log 3)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q297
find_all (answer : ℝ) iff
: (answer = Real.log 27 / Real.log 3)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
