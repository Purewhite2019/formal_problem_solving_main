import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the range of the function $y=\log_2 (\sqrt{\sin x})$ for $0^\circ< x < 180^\circ$?

## Answer
(-\infty, 0]
-/

-- # Formal Answer
noncomputable abbrev Q96.gt := fun (answer : Set ℝ) => answer = ({y : ℝ | y ≤ (0 : ℝ)} : Set ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q96
find (answer : Set ℝ) s.t.
: (answer = { y : ℝ | ∃ x : ℝ, 0 < x ∧ x < Real.pi ∧ y = Real.logb 2 (Real.sqrt (Real.sin x)) })
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q96
find_all (answer : Set ℝ) iff
: (answer = { y : ℝ | ∃ x : ℝ, 0 < x ∧ x < Real.pi ∧ y = Real.logb 2 (Real.sqrt (Real.sin x)) })
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
