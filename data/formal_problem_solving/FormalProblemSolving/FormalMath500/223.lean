import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $-6\leq a \leq -2$ and $3 \leq b \leq 5$, what is the greatest possible value of $\displaystyle\left(a+\frac{1}{b}\right)\left(\frac{1}{b}-a\right) $?  Express your answer as a common fraction.

## Answer
-\frac{35}{9}
-/

-- # Formal Answer
noncomputable abbrev Q223.gt := fun (answer : ℝ) => answer = (-35 / 9 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q223
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {y : ℝ | ∃ a b:ℝ, -6 ≤ a ∧ a ≤ -2 ∧ 3 ≤ b ∧ b ≤ 5 ∧ (y = (a + (1:ℝ) / b) * ((1:ℝ) / b - a))})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q223
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {y : ℝ | ∃ a b:ℝ, -6 ≤ a ∧ a ≤ -2 ∧ 3 ≤ b ∧ b ≤ 5 ∧ (y = (a + (1:ℝ) / b) * ((1:ℝ) / b - a))})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
