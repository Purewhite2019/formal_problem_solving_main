import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $a,$ $b,$ $c,$ and $d$ be positive real numbers such that $a + b + c + d = 10.$  Find the maximum value of $ab^2 c^3 d^4.$

## Answer
27648
-/

-- # Formal Answer
noncomputable abbrev Q304.gt := fun (answer : ℝ) => answer = (27648 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q304
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {x : ℝ | ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧  a + b + c + d = 10 ∧ x = a*(b^2)*(c^3)*(d^4)})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q304
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {x : ℝ | ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧  a + b + c + d = 10 ∧ x = a*(b^2)*(c^3)*(d^4)})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
