import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given the two functions $f(x)=x^3+2x+1$ and $g(x)=x-1$, find $f(g(1))$.

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q141.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q141
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x^3 + 2 * x + 1)
  (h₁ : ∀ x, g x = x - 1)
: (answer = f (g 1))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q141
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x^3 + 2 * x + 1)
  (h₁ : ∀ x, g x = x - 1)
: (answer = f (g 1))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
