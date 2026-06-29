import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the domain of the function $f(x) = \frac{2-x}{\log(2-\log(x-2))}$, where $\log$ is the base $10$ logarithm function? Express your answer in interval notation.

## Answer
(2,12) \cup (12,102)
-/

-- # Formal Answer
noncomputable abbrev Q347.gt := fun (answer : Set ℝ) => answer = ({x : ℝ | (2 : ℝ) < x ∧ x < (102 : ℝ) ∧ x ≠ (12 : ℝ)} : Set ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q347
find (answer : Set ℝ) s.t.
: (answer = {x : ℝ | x - 2 > 0 ∧ 2 - Real.logb 10 (x - 2) > 0 ∧ Real.logb 10 (2 - Real.logb 10 (x - 2)) ≠ 0})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q347
find_all (answer : Set ℝ) iff
: (answer = {x : ℝ | x - 2 > 0 ∧ 2 - Real.logb 10 (x - 2) > 0 ∧ Real.logb 10 (2 - Real.logb 10 (x - 2)) ≠ 0})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
