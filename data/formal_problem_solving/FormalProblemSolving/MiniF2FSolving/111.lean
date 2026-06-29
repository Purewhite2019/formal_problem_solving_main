import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $f (x) = x + 2$ and $g (x) = x^2$, then for what value of $x$ does $f(g(x)) = g(f(x))$? Express your answer as a common fraction.

## Answer
-1/2
-/

-- # Formal Answer
noncomputable abbrev Q111.gt := fun (answer : ℝ) => answer = (-1 / 2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q111
find (answer : ℝ) s.t.
  (x : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x + 2)
  (h₁ : ∀ x, g x = x^2)
  (h₂ : f (g x) = g (f x))
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q111
find_all (answer : ℝ) iff
  (x : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x + 2)
  (h₁ : ∀ x, g x = x^2)
  (h₂ : f (g x) = g (f x))
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
