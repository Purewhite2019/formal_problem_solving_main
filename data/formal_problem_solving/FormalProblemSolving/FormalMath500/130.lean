import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $\sqrt{3x-5}=2$, find all possible values of $x$.

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q130.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q130
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : √(3 * x - 5) = 2)
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q130
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : √(3 * x - 5) = 2)
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
