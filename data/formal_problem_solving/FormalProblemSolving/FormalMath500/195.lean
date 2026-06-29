import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given $f(x) = \frac{\sqrt{x-1}}{x-2}$, what is the smallest possible integer value for $x$ such that $f(x)$ has a real number value?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q195.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q195
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {x:ℝ | x - 2 ≠ 0 ∧ x - 1 ≥ 0})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q195
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {x:ℝ | x - 2 ≠ 0 ∧ x - 1 ≥ 0})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
