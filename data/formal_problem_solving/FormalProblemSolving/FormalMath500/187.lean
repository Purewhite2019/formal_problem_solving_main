import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify the following:$$\frac{3}{\sqrt{27}}$$

## Answer
\frac{\sqrt{3}}{3}
-/

-- # Formal Answer
noncomputable abbrev Q187.gt := fun (answer : ℝ) => answer = ((√(3 : ℝ) : ℝ) / (3 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q187
find (answer : ℝ) s.t.
: (answer = 3 / Real.sqrt 27)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q187
find_all (answer : ℝ) iff
: (answer = 3 / Real.sqrt 27)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
