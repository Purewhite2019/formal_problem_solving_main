import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify: $\frac{\sqrt{2.5^2-0.7^2}}{2.7-2.5}$.

## Answer
12
-/

-- # Formal Answer
noncomputable abbrev Q8.gt := fun (answer : ℝ) => answer = (12 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q8
find (answer : ℝ) s.t.
: ((Real.sqrt (2.5^2 - 0.7^2)) / (2.7 - 2.5) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q8
find_all (answer : ℝ) iff
: ((Real.sqrt (2.5^2 - 0.7^2)) / (2.7 - 2.5) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
