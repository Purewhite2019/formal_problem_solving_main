import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $f(x)=5x^2+3x+4$, what is the value of $f(-2)$?

## Answer
18
-/

-- # Formal Answer
noncomputable abbrev Q118.gt := fun (answer : ℝ) => answer = (18 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q118
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (x : ℝ)
  (h₀ : ∀ x, f x = 5 * x^2 + 3 * x + 4)
: (f (-2) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q118
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (x : ℝ)
  (h₀ : ∀ x, f x = 5 * x^2 + 3 * x + 4)
: (f (-2) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
