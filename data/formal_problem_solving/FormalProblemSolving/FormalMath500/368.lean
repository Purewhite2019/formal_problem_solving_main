import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Consider the function $z(x,y)$ describing the paraboloid
\[z = (2x - y)^2 - 2y^2 - 3y.\]Archimedes and Brahmagupta are playing a game.  Archimedes first chooses $x.$  Afterwards, Brahmagupta chooses $y.$  Archimedes wishes to minimize $z$ while Brahmagupta wishes to maximize $z.$  Assuming that Brahmagupta will play optimally, what value of $x$ should Archimedes choose?

## Answer
-\frac{3}{8}
-/

-- # Formal Answer
noncomputable abbrev Q368.gt := fun (answer : ℝ) => answer = (-3 / 8 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q368
find (answer : ℝ) s.t.
  (y_opt : ℝ)
: (∀ y, (2 * answer - y_opt)^2 - 2 * y_opt^2 - 3 * y_opt ≥ (2 * answer - y)^2 - 2 * y^2 - 3 * y) ∧ (∀ x, ∃ y, (∀ y', (2 * x - y)^2 - 2 * y^2 - 3 * y ≥ (2 * x - y')^2 - 2 * y'^2 - 3 * y') ∧
                      (2 * answer - y_opt)^2 - 2 * y_opt^2 - 3 * y_opt ≤ (2 * x - y)^2 - 2 * y^2 - 3 * y)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q368
find_all (answer : ℝ) iff
  (y_opt : ℝ)
: (∀ y, (2 * answer - y_opt)^2 - 2 * y_opt^2 - 3 * y_opt ≥ (2 * answer - y)^2 - 2 * y^2 - 3 * y) ∧ (∀ x, ∃ y, (∀ y', (2 * x - y)^2 - 2 * y^2 - 3 * y ≥ (2 * x - y')^2 - 2 * y'^2 - 3 * y') ∧
                      (2 * answer - y_opt)^2 - 2 * y_opt^2 - 3 * y_opt ≤ (2 * x - y)^2 - 2 * y^2 - 3 * y)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
