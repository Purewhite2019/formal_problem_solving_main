import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $f(x) = \frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction.

## Answer
\frac{14}{3}
-/

-- # Formal Answer
noncomputable abbrev Q176.gt := fun (answer : ℝ) => answer = (14 / 3 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q176
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = (3 * x - 2) / (x - 2))
: (f (-2) + f (-1) + f (0) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q176
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = (3 * x - 2) / (x - 2))
: (f (-2) + f (-1) + f (0) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
