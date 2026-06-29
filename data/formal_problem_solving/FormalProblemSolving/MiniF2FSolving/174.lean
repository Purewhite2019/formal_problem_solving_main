import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The polynomial $p(x) = x^2+ax+b$ has distinct roots $2a$ and $b$. Find $a+b$.

## Answer
-1
-/

-- # Formal Answer
noncomputable abbrev Q174.gt := fun (answer : ℝ) => answer = (-1 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q174
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = x^2 + a * x + b)
  (h₁ : 2 * a ≠ b)
  (h₂ : f (2 * a) = 0)
  (h₃ : f b = 0)
: (answer = a + b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q174
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = x^2 + a * x + b)
  (h₁ : 2 * a ≠ b)
  (h₂ : f (2 * a) = 0)
  (h₃ : f b = 0)
: (answer = a + b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
