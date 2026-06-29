import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Write $\cfrac{\cfrac{3}{8}+\cfrac{7}{8}}{\cfrac{4}{5}}$ as a simplified fraction.

## Answer
$\cfrac{25}{16}$
-/

-- # Formal Answer
noncomputable abbrev Q124.gt := fun (answer : ℚ) => answer = (25 / 16 : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q124
find (answer : ℚ) s.t.
: (answer = ((3 / 8) + (7 / 8)) / (4 / 5))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q124
find_all (answer : ℚ) iff
: (answer = ((3 / 8) + (7 / 8)) / (4 / 5))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
