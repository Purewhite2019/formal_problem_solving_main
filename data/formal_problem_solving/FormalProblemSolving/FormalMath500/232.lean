import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find all values of $x$ that satisfy the equation $x = \!\sqrt{11-2x} + 4$.

## Answer
x=5
-/

-- # Formal Answer
noncomputable abbrev Q232.gt := fun (answer : ℝ) => answer = (5 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q232
find (answer : ℝ) s.t.
  (x : ℝ)
  (h_eq : x = Real.sqrt (11 - 2 * x) + 4)
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q232
find_all (answer : ℝ) iff
  (x : ℝ)
  (h_eq : x = Real.sqrt (11 - 2 * x) + 4)
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
