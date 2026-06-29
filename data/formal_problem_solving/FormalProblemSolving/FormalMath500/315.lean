import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $a$ and $b$ be positive real numbers such that $ab^2 = 5.$  Find the minimum value of $a^4 + b^8.$

## Answer
50
-/

-- # Formal Answer
noncomputable abbrev Q315.gt := fun (answer : ℝ) => answer = (50 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q315
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {x : ℝ | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b^2 = 5 ∧ a^4 + b^8 = x})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q315
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {x : ℝ | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b^2 = 5 ∧ a^4 + b^8 = x})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
