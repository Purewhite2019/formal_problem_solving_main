import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Add 313.9 to 12.6. Express the result as a decimal.

## Answer
326.5
-/

-- # Formal Answer
noncomputable abbrev Q24.gt := fun (answer : ℝ) => answer = 326.5

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q24
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h₁ : a = 313.9)
  (h₂ : b = 12.6)
: (a + b = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q24
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h₁ : a = 313.9)
  (h₂ : b = 12.6)
: (a + b = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
