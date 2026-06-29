import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A $90^\circ$ rotation around $-4 - 5i$ in the clockwise direction is applied to $3$ (as a complex number).  What is the resulting complex number?

## Answer
1 - 12i
-/

-- # Formal Answer
noncomputable abbrev Q90.gt := fun (answer : ℂ) => answer = ({ re := (1 : ℝ), im := (-12 : ℝ) : ℂ } : ℂ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q90
find (answer : ℂ) s.t.
  (z : ℂ)
  (h_rotation : z = Complex.mk (-4) (-5) + (Complex.mk 3 0 - Complex.mk (-4) (-5)) * Complex.mk (Real.cos (-Real.pi / 2)) (Real.sin (-Real.pi / 2)))
: (z = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q90
find_all (answer : ℂ) iff
  (z : ℂ)
  (h_rotation : z = Complex.mk (-4) (-5) + (Complex.mk 3 0 - Complex.mk (-4) (-5)) * Complex.mk (Real.cos (-Real.pi / 2)) (Real.sin (-Real.pi / 2)))
: (z = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
