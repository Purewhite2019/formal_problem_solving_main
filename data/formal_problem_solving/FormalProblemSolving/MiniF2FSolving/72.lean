import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $f(x)=4x^3+1$, find $f^{-1}(33)$.

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q72.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q72
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = 4 * x^3 + 1)
  (h_inv : Function.Bijective f)
: (answer = Function.invFun f 33)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q72
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = 4 * x^3 + 1)
  (h_inv : Function.Bijective f)
: (answer = Function.invFun f 33)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
