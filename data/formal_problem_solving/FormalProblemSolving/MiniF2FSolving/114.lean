import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given that $x+y = 7$ and $3x+y = 45,$ evaluate $x^2-y^2.$

## Answer
217
-/

-- # Formal Answer
noncomputable abbrev Q114.gt := fun (answer : ℝ) => answer = (217 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q114
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : x + y = 7)
  (h₁ : 3 * x + y = 45)
: (answer = x^2 - y^2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q114
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : x + y = 7)
  (h₁ : 3 * x + y = 45)
: (answer = x^2 - y^2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
