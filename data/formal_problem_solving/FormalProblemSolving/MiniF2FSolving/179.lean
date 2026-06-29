import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Three plus the reciprocal of a number equals 7 divided by that number.  What is the number?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q179.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q179
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : 3 + 1 / x = 7 / x)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q179
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : 3 + 1 / x = 7 / x)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
