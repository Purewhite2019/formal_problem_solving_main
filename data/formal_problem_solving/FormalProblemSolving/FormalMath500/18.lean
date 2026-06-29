import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $3x + 2(1 + x) = 17$, what is the value of $6x + 5$?

## Answer
23
-/

-- # Formal Answer
noncomputable abbrev Q18.gt := fun (answer : ℝ) => answer = (23 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q18
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : 3 * x + 2 * (1 + x) = 17)
: (6 * x + 5 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q18
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : 3 * x + 2 * (1 + x) = 17)
: (6 * x + 5 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
