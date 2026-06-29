import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the value of $\sqrt{1,\!000,\!000} - \sqrt[3]{1,\!000,\!000}$?

## Answer
900
-/

-- # Formal Answer
noncomputable abbrev Q216.gt := fun (answer : ℝ) => answer = (900 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q216
find (answer : ℝ) s.t.
: (answer = Real.sqrt 1000000 - (1000000 : ℝ) ^ (1/3 : ℝ))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q216
find_all (answer : ℝ) iff
: (answer = Real.sqrt 1000000 - (1000000 : ℝ) ^ (1/3 : ℝ))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
