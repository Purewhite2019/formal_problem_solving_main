import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify the following expression in $x$: $$19x + 1 - 4x - 81.$$

## Answer
15x - 80
-/

-- # Formal Answer
noncomputable abbrev Q62.gt := fun (x : ℝ) (answer : ℝ) => answer = (((15 : ℝ) * x : ℝ) - (80 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q62
  (x : ℝ)
find (answer : ℝ) s.t.
: (answer = 19 * x + 1 - 4 * x - 81)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q62
  (x : ℝ)
find_all (answer : ℝ) iff
: (answer = 19 * x + 1 - 4 * x - 81)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
