import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction.

## Answer
\frac{11}{2}
-/

-- # Formal Answer
noncomputable abbrev Q210.gt := fun (answer : ℝ) => answer = ((↑(11 : ℚ) : ℝ) / (2 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q210
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {x:ℝ | |x + 5| - |3 * x - 6| = 0})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q210
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {x:ℝ | |x + 5| - |3 * x - 6| = 0})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
