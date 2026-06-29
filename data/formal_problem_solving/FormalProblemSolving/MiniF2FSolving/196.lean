import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose that $h(x)=f^{-1}(x)$. If $h(2)=10$, $h(10)=1$ and $h(1)=2$, what is $f(f(10))$?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q196.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q196
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h : ℝ → ℝ)
  (h_inv : ∀ x, h (f x) = x)
  (h_inv' : ∀ x, f (h x) = x)
  (h₀ : h 2 = 10)
  (h₁ : h 10 = 1)
  (h₂ : h 1 = 2)
: (answer = f (f 10))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q196
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h : ℝ → ℝ)
  (h_inv : ∀ x, h (f x) = x)
  (h_inv' : ∀ x, f (h x) = x)
  (h₀ : h 2 = 10)
  (h₁ : h 10 = 1)
  (h₂ : h 1 = 2)
: (answer = f (f 10))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
