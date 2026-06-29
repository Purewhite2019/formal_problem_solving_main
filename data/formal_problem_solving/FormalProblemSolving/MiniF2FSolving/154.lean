import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $f(x)=x^3-8$, what is $f^{-1}(f(f^{-1}(19)))$?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q154.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q154
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (f_inv : ℝ → ℝ)
  (h_f : ∀ x, f x = x^3 - 8)
  (h_f_inv : Function.LeftInverse f_inv f)
: (answer = f_inv (f (f_inv 19)))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q154
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (f_inv : ℝ → ℝ)
  (h_f : ∀ x, f x = x^3 - 8)
  (h_f_inv : Function.LeftInverse f_inv f)
: (answer = f_inv (f (f_inv 19)))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
