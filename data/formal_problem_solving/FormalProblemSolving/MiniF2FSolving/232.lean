import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x) = 2x-3$ and $g(x) = x+1$. What is the value of $g(f(5)-1)$?

## Answer
7
-/

-- # Formal Answer
noncomputable abbrev Q232.gt := fun (answer : ℝ) => answer = (7 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q232
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f x = 2 * x - 3)
  (h₁ : ∀ x, g x = x + 1)
: (answer = g (f 5 - 1))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q232
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f x = 2 * x - 3)
  (h₁ : ∀ x, g x = x + 1)
: (answer = g (f 5 - 1))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
