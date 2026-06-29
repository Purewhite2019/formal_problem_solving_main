import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If two (positive) prime numbers are roots of the equation $x^2-12x+k=0$, what is the value of $k$?

## Answer
35
-/

-- # Formal Answer
noncomputable abbrev Q176.gt := fun (answer : ℝ) => answer = (35 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q176
find (answer : ℝ) s.t.
  (m : ℕ)
  (n : ℕ)
  (k : ℝ)
  (f : ℝ → ℝ)
  (h₀ : Nat.Prime m)
  (h₁ : Nat.Prime n)
  (h₂ : ∀ x, f x = x^2 - 12 * x + k)
  (h₃ : f m = 0)
  (h₄ : f n = 0)
  (h₅ : m ≠ n)
: (answer = k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q176
find_all (answer : ℝ) iff
  (m : ℕ)
  (n : ℕ)
  (k : ℝ)
  (f : ℝ → ℝ)
  (h₀ : Nat.Prime m)
  (h₁ : Nat.Prime n)
  (h₂ : ∀ x, f x = x^2 - 12 * x + k)
  (h₃ : f m = 0)
  (h₄ : f n = 0)
  (h₅ : m ≠ n)
: (answer = k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
