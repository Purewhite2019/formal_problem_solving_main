import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Evaluate $\log_{5^2}5^4$.

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q103.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q103
find (answer : ℝ) s.t.
: (answer = Real.logb (5^2) (5^4))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q103
find_all (answer : ℝ) iff
: (answer = Real.logb (5^2) (5^4))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
