import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $3a + b + c = -3, a+3b+c = 9, a+b+3c = 19$, then find $abc$.

## Answer
-56
-/

-- # Formal Answer
noncomputable abbrev Q375.gt := fun (answer : ℝ) => answer = (-56 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q375
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : 3 * a + b + c = -3)
  (h₁ : a + 3 * b + c = 9)
  (h₂ : a + b + 3 * c = 19)
: (answer = a * b * c)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q375
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : 3 * a + b + c = -3)
  (h₁ : a + 3 * b + c = 9)
  (h₂ : a + b + 3 * c = 19)
: (answer = a * b * c)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
