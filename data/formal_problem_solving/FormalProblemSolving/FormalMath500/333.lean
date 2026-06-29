import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $a$ be a positive real number such that all the roots of
\[x^3 + ax^2 + ax + 1 = 0\]are real.  Find the smallest possible value of $a.$

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q333.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q333
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {a : ℝ | 0 < a ∧ a^4 - 8 * a^3 + 18 * a^2 - 27 ≥ 0})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q333
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {a : ℝ | 0 < a ∧ a^4 - 8 * a^3 + 18 * a^2 - 27 ≥ 0})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
