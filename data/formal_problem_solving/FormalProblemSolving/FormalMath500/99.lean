import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute $\arcsin \left( -\frac{1}{2} \right).$  Express your answer in radians.

## Answer
-\frac{\pi}{6}
-/

-- # Formal Answer
noncomputable abbrev Q99.gt := fun (answer : ℝ) => answer = ((-(π : ℝ) : ℝ) / (6 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q99
find (answer : ℝ) s.t.
: (answer = Real.arcsin (-1/2))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q99
find_all (answer : ℝ) iff
: (answer = Real.arcsin (-1/2))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
