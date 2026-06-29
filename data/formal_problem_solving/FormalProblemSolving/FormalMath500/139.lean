import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the largest value of $x$ that satisfies the equation $|5x-1|=x+3$.

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q139.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q139
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {x : ℝ | abs (5 * x - 1) = x + 3})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q139
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {x : ℝ | abs (5 * x - 1) = x + 3})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
