import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For what value of $c$ will the circle with equation $x^2 - 10x + y^2 + 6y + c = 0$ have a radius of length 1?

## Answer
33
-/

-- # Formal Answer
noncomputable abbrev Q215.gt := fun (answer : ℝ) => answer = (33 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q215
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : ∀ x y, x^2 - 10 * x + y^2 + 6 * y + c = (x - a)^2 + (y + b)^2 - 1)
: (c = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q215
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : ∀ x y, x^2 - 10 * x + y^2 + 6 * y + c = (x - a)^2 + (y + b)^2 - 1)
: (c = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
