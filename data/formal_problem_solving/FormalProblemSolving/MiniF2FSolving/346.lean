import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the units digit of $29 \cdot 79 + 31 \cdot 81$.

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q346.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q346
find (answer : ℕ) s.t.
: (answer = (29 * 79 + 31 * 81) % 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q346
find_all (answer : ℕ) iff
: (answer = (29 * 79 + 31 * 81) % 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
