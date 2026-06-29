import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $f(x)=ax^4-bx^2+x+5$ and $f(-3)=2,$ then what is the value of $f(3)$?

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q325.gt := fun (answer : ℝ) => answer = (8 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q325
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = a * x^4 - b * x^2 + x + 5)
  (h₂ : f (-3) = 2)
: (answer = f 3)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q325
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = a * x^4 - b * x^2 + x + 5)
  (h₂ : f (-3) = 2)
: (answer = f 3)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
