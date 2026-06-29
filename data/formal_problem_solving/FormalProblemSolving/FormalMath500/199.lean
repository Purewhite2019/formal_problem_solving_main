import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the maximum value of $4(x + 7)(2 - x)$, over all real numbers $x$?

## Answer
81
-/

-- # Formal Answer
noncomputable abbrev Q199.gt := fun (answer : ℝ) => answer = (81 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q199
find (answer : ℝ) s.t.
  (x : ℝ)
: (IsGreatest {y : ℝ| ∃ x : ℝ, y = 4 * (x + 7) * (2 - x)} answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q199
find_all (answer : ℝ) iff
  (x : ℝ)
: (IsGreatest {y : ℝ| ∃ x : ℝ, y = 4 * (x + 7) * (2 - x)} answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
