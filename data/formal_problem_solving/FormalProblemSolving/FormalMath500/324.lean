import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $a,$ $b,$ and $c$ be positive real numbers.  Find the set of all possible values of
\[\frac{c}{a} + \frac{a}{b + c} + \frac{b}{c}.\]

## Answer
(2,\infty)
-/

-- # Formal Answer
noncomputable abbrev Q324.gt := fun (answer : Set ℝ) => answer = ({x : ℝ | x > (2 : ℝ)} : Set ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q324
find (answer : Set ℝ) s.t.
: (answer = {x : ℝ | ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ x = c / a + a / (b + c) + b / c})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q324
find_all (answer : Set ℝ) iff
: (answer = {x : ℝ | ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ x = c / a + a / (b + c) + b / c})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
