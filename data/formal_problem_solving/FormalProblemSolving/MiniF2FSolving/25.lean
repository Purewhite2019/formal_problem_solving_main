import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the remainder of 2004 divided by 12?

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q25.gt := fun (answer : ℕ) => answer = (0 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q25
find (answer : ℕ) s.t.
: (answer = 2004 % 12)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q25
find_all (answer : ℕ) iff
: (answer = 2004 % 12)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
