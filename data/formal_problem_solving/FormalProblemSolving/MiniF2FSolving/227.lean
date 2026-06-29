import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose $f(x)$ is an invertible function, and suppose that $f(2)=f^{-1}(2)=4$.

What is the value of $f(f(2))$?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q227.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q227
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (f_inv : ℝ → ℝ)
  (h_invertible : Function.Bijective f)
  (h_inv : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x)
  (h_f2 : f 2 = 4)
  (h_finv2 : f_inv 2 = 4)
: (answer = f (f 2))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q227
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (f_inv : ℝ → ℝ)
  (h_invertible : Function.Bijective f)
  (h_inv : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x)
  (h_f2 : f 2 = 4)
  (h_finv2 : f_inv 2 = 4)
: (answer = f (f 2))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
