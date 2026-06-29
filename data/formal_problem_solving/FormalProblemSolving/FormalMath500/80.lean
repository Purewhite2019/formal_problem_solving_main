import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify $\tan 100^\circ + 4 \sin 100^\circ.$

## Answer
-\sqrt{3}
-/

-- # Formal Answer
noncomputable abbrev Q80.gt := fun (answer : ℝ) => answer = (-(√(3 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q80
find (answer : ℝ) s.t.
: (answer = Real.tan (100 * Real.pi / 180) + 4 * Real.sin (100 * Real.pi / 180))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q80
find_all (answer : ℝ) iff
: (answer = Real.tan (100 * Real.pi / 180) + 4 * Real.sin (100 * Real.pi / 180))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
