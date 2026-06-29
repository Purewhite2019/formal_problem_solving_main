import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Three plus the reciprocal of a number equals 7 divided by that number.  What is the number?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q217.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q217
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : 3 + ((1:ℝ) / x) = (7:ℝ) / x)
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q217
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : 3 + ((1:ℝ) / x) = (7:ℝ) / x)
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
