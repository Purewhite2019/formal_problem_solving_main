import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Evaluate: $\left( \frac{1}{2} + \frac{1}{3} \right) \left( \frac{1}{2} - \frac{1}{3} \right)$

## Answer
\frac{5}{36}
-/

-- # Formal Answer
noncomputable abbrev Q16.gt := fun (answer : ℚ) => answer = (5 / 36 : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q16
find (answer : ℚ) s.t.
: (answer = ((1 : ℚ)/ 2 + 1 / 3) * (1 / 2 - 1 / 3))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q16
find_all (answer : ℚ) iff
: (answer = ((1 : ℚ)/ 2 + 1 / 3) * (1 / 2 - 1 / 3))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
