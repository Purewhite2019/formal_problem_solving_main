import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the units digit of $16^{17} \times 17^{18} \times 18^{19}$.

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q308.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q308
find (answer : ℕ) s.t.
: (answer = (16^17 * 17^18 * 18^19) % 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q308
find_all (answer : ℕ) iff
: (answer = (16^17 * 17^18 * 18^19) % 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
