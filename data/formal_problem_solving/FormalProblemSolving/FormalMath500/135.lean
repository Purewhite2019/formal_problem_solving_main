import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify the following expression to a simplified fraction: $$\sqrt{\dfrac{\dfrac{5}{\sqrt{80}}+\dfrac{\sqrt{845}}{9}+\sqrt{45}}{\sqrt5}}.$$

## Answer
\frac{13}{6}
-/

-- # Formal Answer
noncomputable abbrev Q135.gt := fun (answer : ℝ) => answer = ((↑(13 : ℚ) : ℝ) / (6 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q135
find (answer : ℝ) s.t.
: (Real.sqrt ((5 / Real.sqrt 80 + Real.sqrt 845 / 9 + Real.sqrt 45) / Real.sqrt 5) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q135
find_all (answer : ℝ) iff
: (Real.sqrt ((5 / Real.sqrt 80 + Real.sqrt 845 / 9 + Real.sqrt 45) / Real.sqrt 5) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
