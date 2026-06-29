import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A $90^\circ$ rotation around the origin in the counter-clockwise direction is applied to $7 + 2i.$  What is the resulting complex number?

## Answer
-2 + 7i
-/

-- # Formal Answer
noncomputable abbrev Q89.gt := fun (answer : ℂ) => answer = ((-2 : ℂ) + ((7 : ℂ) * (Complex.I : ℂ) : ℂ) : ℂ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q89
find (answer : ℂ) s.t.
  (z : ℂ)
  (z_rotated : ℂ)
  (h₀ : z = 7 + 2 * Complex.I)
  (h₁ : z_rotated = Complex.I * z)
: (z_rotated = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q89
find_all (answer : ℂ) iff
  (z : ℂ)
  (z_rotated : ℂ)
  (h₀ : z = 7 + 2 * Complex.I)
  (h₁ : z_rotated = Complex.I * z)
: (z_rotated = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
