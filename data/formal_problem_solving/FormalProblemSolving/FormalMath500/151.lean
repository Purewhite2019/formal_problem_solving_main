import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $a = 8$, what is the value of $\left(16\sqrt[3]{a^2}\right)^{\frac 13}$?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q151.gt := fun (answer : ℝ) => answer = (4 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q151
find (answer : ℝ) s.t.
  (a : ℝ)
  (h₀ : a = 8)
: ((16 * (a^2) ^ ((1:ℝ)/3)) ^ ((1:ℝ)/3) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q151
find_all (answer : ℝ) iff
  (a : ℝ)
  (h₀ : a = 8)
: ((16 * (a^2) ^ ((1:ℝ)/3)) ^ ((1:ℝ)/3) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
