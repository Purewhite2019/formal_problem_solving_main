import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many $y$-intercepts does the graph of the parabola $x = y^2 - 4y - 1$ have?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q115.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q115
find (answer : ℕ) s.t.
  (y_intercepts : Set ℝ)
  (h₀ : y_intercepts = {y : ℝ | 0 = y^2 - 4*y - 1})
: (y_intercepts.encard = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q115
find_all (answer : ℕ) iff
  (y_intercepts : Set ℝ)
  (h₀ : y_intercepts = {y : ℝ | 0 = y^2 - 4*y - 1})
: (y_intercepts.encard = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
