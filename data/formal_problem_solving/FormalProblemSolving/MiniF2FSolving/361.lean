import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Evaluate $\left(\frac{i}{2}\right)^2$.

## Answer
-\frac{1}{4}
-/

-- # Formal Answer
noncomputable abbrev Q361.gt := fun (answer : ℂ) => answer = (-(1 / 4 : ℂ) : ℂ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q361
find (answer : ℂ) s.t.
: (answer = (Complex.I / 2)^2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q361
find_all (answer : ℂ) iff
: (answer = (Complex.I / 2)^2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
