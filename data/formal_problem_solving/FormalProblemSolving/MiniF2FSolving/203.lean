import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the value of $-a-b^2+3ab$ if $a=-1$ and $b=5$?

## Answer
-39
-/

-- # Formal Answer
noncomputable abbrev Q203.gt := fun (answer : ℝ) => answer = (-39 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q203
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h₀ : a = -1)
  (h₁ : b = 5)
: (answer = -a - b^2 + 3 * (a * b))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q203
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h₀ : a = -1)
  (h₁ : b = 5)
: (answer = -a - b^2 + 3 * (a * b))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
