import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Bill walks $\frac{1}{2}$ mile south, then $\frac{3}{4}$ mile east, and finally $\frac{1}{2}$ mile south. How many miles is he, in a direct line, from his starting point?  Express your answer as a decimal to the nearest hundredth.

## Answer
1.25
-/

-- # Formal Answer
noncomputable abbrev Q2.gt := fun (answer : ℝ) => answer = 1.25

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q2
find (answer : ℝ) s.t.
: (answer = Real.sqrt ((1/2 + 1/2)^2 + (3/4)^2))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q2
find_all (answer : ℝ) iff
: (answer = Real.sqrt ((1/2 + 1/2)^2 + (3/4)^2))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
