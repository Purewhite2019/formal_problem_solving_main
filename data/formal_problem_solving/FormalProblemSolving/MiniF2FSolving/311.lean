import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given $f(x)=cx^3-9x+3$ and $f(2)=9$, find the value of $c$.

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q311.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q311
find (answer : ℝ) s.t.
  (c : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = c * x^3 - 9 * x + 3)
  (h₁ : f 2 = 9)
: (answer = c)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q311
find_all (answer : ℝ) iff
  (c : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = c * x^3 - 9 * x + 3)
  (h₁ : f 2 = 9)
: (answer = c)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
