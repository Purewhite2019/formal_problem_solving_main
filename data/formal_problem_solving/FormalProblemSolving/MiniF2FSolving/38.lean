import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the positive value of the expression $\sqrt{x^3 - 2^y}$ when $x = 5$ and $y = 2$?

## Answer
11
-/

-- # Formal Answer
noncomputable abbrev Q38.gt := fun (answer : ℝ) => answer = (11 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q38
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : x = 5)
  (h₁ : y = 2)
: (answer = Real.sqrt (x^3 - 2^y))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q38
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : x = 5)
  (h₁ : y = 2)
: (answer = Real.sqrt (x^3 - 2^y))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
