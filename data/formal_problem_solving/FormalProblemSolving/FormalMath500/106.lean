import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A curve is parameterized by
\[(x,y) = (t^3 + 7, -3t^2 - 6t - 5).\]Find the point the curve passes through at $t = 2.$

## Answer
(15,-29)
-/

-- # Formal Answer
noncomputable abbrev Q106.gt := fun (answer : ℝ × ℝ) => answer = (((15 : ℝ), (-29 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q106
find (answer : ℝ × ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : x = 2^3 + 7)
  (h₁ : y = -3 * 2^2 - 6 * 2 - 5)
: ((x, y) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q106
find_all (answer : ℝ × ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : x = 2^3 + 7)
  (h₁ : y = -3 * 2^2 - 6 * 2 - 5)
: ((x, y) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
