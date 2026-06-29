import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve
\[\frac{1}{x - 5} > 0.\]Enter your answer using interval notation.

## Answer
(5,\infty)
-/

-- # Formal Answer
noncomputable abbrev Q375.gt := fun (answer : Set ℝ) => answer = ({x : ℝ | x > (5 : ℝ)} : Set ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q375
find (answer : Set ℝ) s.t.
: (answer = {x:ℝ | 1 / (x - 5) > 0})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q375
find_all (answer : Set ℝ) iff
: (answer = {x:ℝ | 1 / (x - 5) > 0})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
