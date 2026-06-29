import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x) = x^2 - 4\sqrt{x} + 1$. What is $f(f(4))$?

## Answer
70
-/

-- # Formal Answer
noncomputable abbrev Q122.gt := fun (answer : ℝ) => answer = (70 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q122
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = x^2 - 4 * Real.sqrt x + 1)
: (answer = f (f 4))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q122
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = x^2 - 4 * Real.sqrt x + 1)
: (answer = f (f 4))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
