import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?

## Answer
42
-/

-- # Formal Answer
noncomputable abbrev Q25.gt := fun (answer : ℕ) => answer = (42 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q25
find (answer : ℕ) s.t.
  (triangle_perimeter : ℕ)
  (hexagon_perimeter : ℕ)
  (h_triangle : triangle_perimeter = 21)
  (h_hexagon : hexagon_perimeter = 2 * triangle_perimeter)
: (hexagon_perimeter = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q25
find_all (answer : ℕ) iff
  (triangle_perimeter : ℕ)
  (hexagon_perimeter : ℕ)
  (h_triangle : triangle_perimeter = 21)
  (h_hexagon : hexagon_perimeter = 2 * triangle_perimeter)
: (hexagon_perimeter = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
