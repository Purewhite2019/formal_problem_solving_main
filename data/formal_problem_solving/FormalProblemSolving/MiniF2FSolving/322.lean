import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $f(x) = x + 1$ and $g(x) = x^2 + 3$, what is the value of $f(g(2))$?

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q322.gt := fun (answer : ℝ) => answer = (8 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q322
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x + 1)
  (h₁ : ∀ x, g x = x^2 + 3)
: (answer = f (g 2))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q322
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x + 1)
  (h₁ : ∀ x, g x = x^2 + 3)
: (answer = f (g 2))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
