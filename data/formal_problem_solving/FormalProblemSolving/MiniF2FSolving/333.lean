import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the modulo 4 residue of $121 \cdot 122 \cdot 123$.

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q333.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q333
find (answer : ℕ) s.t.
: (answer = (121 * 122 * 123) % 4)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q333
find_all (answer : ℕ) iff
: (answer = (121 * 122 * 123) % 4)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
