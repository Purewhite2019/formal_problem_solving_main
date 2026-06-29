import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Evaluate $\left\lceil\sqrt{27}\right\rceil - \left\lfloor\sqrt{26}\right\rfloor$.

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q75.gt := fun (answer : ℤ) => answer = (1 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q75
find (answer : ℤ) s.t.
: (answer = Int.ceil (Real.sqrt 27) - Int.floor (Real.sqrt 26))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q75
find_all (answer : ℤ) iff
: (answer = Int.ceil (Real.sqrt 27) - Int.floor (Real.sqrt 26))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
