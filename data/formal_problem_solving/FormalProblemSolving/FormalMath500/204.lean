import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the 2003rd term of the sequence of odd numbers 1, 3, 5, 7, $\dots$?

## Answer
4005
-/

-- # Formal Answer
noncomputable abbrev Q204.gt := fun (answer : ℕ) => answer = (4005 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q204
find (answer : ℕ) s.t.
: (answer = (fun n : ℕ => 2 * n + 1) 2002)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q204
find_all (answer : ℕ) iff
: (answer = (fun n : ℕ => 2 * n + 1) 2002)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
