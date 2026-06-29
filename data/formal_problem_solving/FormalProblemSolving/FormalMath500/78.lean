import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find $\sin 20^\circ \sin 40^\circ \sin 60^\circ \sin 80^\circ \sin 100^\circ \sin 120^\circ \sin 140^\circ \sin 160^\circ.$

## Answer
\frac{9}{256}
-/

-- # Formal Answer
noncomputable abbrev Q78.gt := fun (answer : ℝ) => answer = (9 / 256 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q78
find (answer : ℝ) s.t.
: (Real.sin (20 * Real.pi / 180) * Real.sin (40 * Real.pi / 180) * Real.sin (60 * Real.pi / 180) * Real.sin (80 * Real.pi / 180) * Real.sin (100 * Real.pi / 180) * Real.sin (120 * Real.pi / 180) * Real.sin (140 * Real.pi / 180) * Real.sin (160 * Real.pi / 180) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q78
find_all (answer : ℝ) iff
: (Real.sin (20 * Real.pi / 180) * Real.sin (40 * Real.pi / 180) * Real.sin (60 * Real.pi / 180) * Real.sin (80 * Real.pi / 180) * Real.sin (100 * Real.pi / 180) * Real.sin (120 * Real.pi / 180) * Real.sin (140 * Real.pi / 180) * Real.sin (160 * Real.pi / 180) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
