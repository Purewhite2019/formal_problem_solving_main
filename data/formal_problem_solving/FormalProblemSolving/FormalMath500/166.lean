import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose that $f$ is a function and $f^{-1}$ is the inverse of $f$.  If $f(1)=2$, $f(2) = 6$, and $f(3)=5$, then what is $f^{-1}(f^{-1}(6))$?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q166.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q166
find (answer : ℕ) s.t.
  (f : ℕ → ℕ)
  (f_inv : ℕ → ℕ)
  (h_inv : ∀ x, f_inv (f x) = x)
  (h₁ : f 1 = 2)
  (h₂ : f 2 = 6)
  (h₃ : f 3 = 5)
: (f_inv (f_inv 6) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q166
find_all (answer : ℕ) iff
  (f : ℕ → ℕ)
  (f_inv : ℕ → ℕ)
  (h_inv : ∀ x, f_inv (f x) = x)
  (h₁ : f 1 = 2)
  (h₂ : f 2 = 6)
  (h₃ : f 3 = 5)
: (f_inv (f_inv 6) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
