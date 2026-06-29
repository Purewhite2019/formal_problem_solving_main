import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $a = 8$, what is the value of $\left(16\sqrt[3]{a^2}\right)^{\frac 13}$?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q320.gt := fun (answer : ℝ) => answer = (4 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q320
find (answer : ℝ) s.t.
  (a : ℝ)
  (h₀ : a = 8)
: (answer = (16 * (a^2) ^ (1/3 : ℝ)) ^ (1/3 : ℝ))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q320
find_all (answer : ℝ) iff
  (a : ℝ)
  (h₀ : a = 8)
: (answer = (16 * (a^2) ^ (1/3 : ℝ)) ^ (1/3 : ℝ))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
