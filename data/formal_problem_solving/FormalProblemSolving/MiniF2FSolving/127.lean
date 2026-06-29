import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the modulo 4 residue of $17 \cdot 18$.

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q127.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q127
find (answer : ℕ) s.t.
: (answer = (17 * 18) % 4)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q127
find_all (answer : ℕ) iff
: (answer = (17 * 18) % 4)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
