import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the maximum value of
\[f(x,y) = x \sqrt{1 - y^2} + y \sqrt{1 - x^2},\]where $-1 \le x,$ $y \le 1.$

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q327.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q327
find (answer : ℝ) s.t.
: ((∀ (a b : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ -1 ≤ b ∧ b ≤ 1 → a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) ≤ answer) ∧
  (∃ (a b : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ -1 ≤ b ∧ b ≤ 1 ∧ a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) = answer))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q327
find_all (answer : ℝ) iff
: ((∀ (a b : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ -1 ≤ b ∧ b ≤ 1 → a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) ≤ answer) ∧
  (∃ (a b : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ -1 ≤ b ∧ b ≤ 1 ∧ a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) = answer))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
