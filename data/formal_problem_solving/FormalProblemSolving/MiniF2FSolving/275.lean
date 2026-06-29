import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The expression $10x^2-x-24$ can be written as $(Ax-8)(Bx+3),$ where $A$ and $B$ are integers. What is $AB + B$?

## Answer
12
-/

-- # Formal Answer
noncomputable abbrev Q275.gt := fun (answer : ℤ) => answer = (12 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q275
find (answer : ℤ) s.t.
  (a : ℤ)
  (b : ℤ)
  (h₀ : ∀ x : ℝ, 10 * x^2 - x - 24 = (a * x - 8) * (b * x + 3))
: (answer = a * b + b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q275
find_all (answer : ℤ) iff
  (a : ℤ)
  (b : ℤ)
  (h₀ : ∀ x : ℝ, 10 * x^2 - x - 24 = (a * x - 8) * (b * x + 3))
: (answer = a * b + b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
