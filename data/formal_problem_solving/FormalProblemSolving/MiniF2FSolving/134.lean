import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose that $a$ and $b$ are nonzero real numbers, and that the equation  $${x^2 + ax + b = 0}$$ has solutions $a$ and $b$. Then what is the pair $(a,b)$?

## Answer
(1, -2)
-/

-- # Formal Answer
noncomputable abbrev Q134.gt := fun (answer : ℝ × ℝ) => answer = (((1 : ℝ), (-2 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q134
find (answer : ℝ × ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁ : a ≠ b)
  (h₂ : ∀ x, f x = x^2 + a * x + b)
  (h₃ : f a = 0)
  (h₄ : f b = 0)
: (answer = (a, b))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q134
find_all (answer : ℝ × ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁ : a ≠ b)
  (h₂ : ∀ x, f x = x^2 + a * x + b)
  (h₃ : f a = 0)
  (h₄ : f b = 0)
: (answer = (a, b))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
