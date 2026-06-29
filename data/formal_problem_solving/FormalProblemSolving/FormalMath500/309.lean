import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $a,$ $b,$ $c,$ $d$ be positive real numbers such that $a + b + c + d = 1.$  Find the minimum value of
\[\frac{1}{a} + \frac{1}{b} + \frac{4}{c} + \frac{16}{d}.\]

## Answer
64
-/

-- # Formal Answer
noncomputable abbrev Q309.gt := fun (answer : ℝ) => answer = (64 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q309
find (answer : ℝ) s.t.
: (IsLeast {x : ℝ | ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d = 1 ∧ 1 / a + 1 / b + 4 / c + 16 / d = x} answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q309
find_all (answer : ℝ) iff
: (IsLeast {x : ℝ | ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d = 1 ∧ 1 / a + 1 / b + 4 / c + 16 / d = x} answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
