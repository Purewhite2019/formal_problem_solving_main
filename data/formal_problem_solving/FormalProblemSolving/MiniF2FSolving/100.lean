import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find $\frac{1}{a-1}+\frac{1}{b-1},$ where $a$ and $b$ are the roots of the quadratic equation $2x^2-7x+2 = 0.$

## Answer
-1
-/

-- # Formal Answer
noncomputable abbrev Q100.gt := fun (answer : ℝ) => answer = (-1 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q100
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = 2 * x^2 - 7 * x + 2)
  (h₁ : f a = 0)
  (h₂ : f b = 0)
  (h₃ : a ≠ b)
: (answer = 1 / (a - 1) + 1 / (b - 1))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q100
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = 2 * x^2 - 7 * x + 2)
  (h₁ : f a = 0)
  (h₂ : f b = 0)
  (h₃ : a ≠ b)
: (answer = 1 / (a - 1) + 1 / (b - 1))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
