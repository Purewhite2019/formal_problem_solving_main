import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the $x$-coordinate for the $x$-intercept of the line containing the points $(7,4)$ and $(6,3)$?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q82.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q82
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = a * x + b)
  (h₁ : f 7 = 4)
  (h₂ : f 6 = 3)
: (f answer = 0)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q82
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = a * x + b)
  (h₁ : f 7 = 4)
  (h₂ : f 6 = 3)
: (f answer = 0)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
