import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute $91^2$ in your head.

## Answer
8281
-/

-- # Formal Answer
noncomputable abbrev Q299.gt := fun (answer : ℕ) => answer = (8281 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q299
find (answer : ℕ) s.t.
: (answer = 91^2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q299
find_all (answer : ℕ) iff
: (answer = 91^2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
