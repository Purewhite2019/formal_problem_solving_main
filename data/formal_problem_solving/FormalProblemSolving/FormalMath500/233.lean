import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify and write the result with a rational denominator: $$\sqrt{\sqrt[3]{\sqrt{\frac{1}{729}}}}$$

## Answer
\frac{\sqrt{3}}{3}
-/

-- # Formal Answer
noncomputable abbrev Q233.gt := fun (answer : ℝ) => answer = ((√(3 : ℝ) : ℝ) / (3 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q233
find (answer : ℝ) s.t.
: (answer = Real.sqrt (Real.rpow (Real.sqrt ((1:ℝ) / 729)) ((1:ℝ) / 3)))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q233
find_all (answer : ℝ) iff
: (answer = Real.sqrt (Real.rpow (Real.sqrt ((1:ℝ) / 729)) ((1:ℝ) / 3)))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
