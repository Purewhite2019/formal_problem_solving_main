import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the remainder when $129^{34}+96^{38}$ is divided by $11$?

## Answer
9
-/

-- # Formal Answer
noncomputable abbrev Q241.gt := fun (answer : ℕ) => answer = (9 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q241
find (answer : ℕ) s.t.
: (answer = (129^34 + 96^38) % 11)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q241
find_all (answer : ℕ) iff
: (answer = (129^34 + 96^38) % 11)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
