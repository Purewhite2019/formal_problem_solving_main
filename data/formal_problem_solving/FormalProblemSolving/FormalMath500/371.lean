import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $a,$ $b,$ $c$ be real numbers such that
\[|ax^2 + bx + c| \le 1\]for all $0 \le x \le 1.$  Find the largest possible value of $|a| + |b| + |c|.$

## Answer
17
-/

-- # Formal Answer
noncomputable abbrev Q371.gt := fun (answer : ℝ) => answer = (17 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q371
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {y : ℝ | ∃ a b c : ℝ, y = |a| + |b| + |c| ∧ (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → |a * x^2 + b * x + c| ≤ 1)})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q371
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {y : ℝ | ∃ a b c : ℝ, y = |a| + |b| + |c| ∧ (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → |a * x^2 + b * x + c| ≤ 1)})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
