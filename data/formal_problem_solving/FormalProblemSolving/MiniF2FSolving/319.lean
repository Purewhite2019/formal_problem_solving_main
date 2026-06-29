import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The function $f$, defined on the set of ordered pairs of positive integers, satisfies the following properties:
$ f(x, x) = x,\; f(x, y) = f(y, x), {\rm \ and\ } (x+y)f(x, y) = yf(x, x+y). $
Calculate $f(14,52)$.

## Answer
364
-/

-- # Formal Answer
noncomputable abbrev Q319.gt := fun (answer : ℝ) => answer = (364 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q319
find (answer : ℝ) s.t.
  (f : ℕ → ℕ → ℝ)
  (h₀ : ∀ x, 0 < x → f x x = x)
  (h₁ : ∀ x y, (0 < x ∧ 0 < y) → f x y = f y x)
  (h₂ : ∀ x y, (0 < x ∧ 0 < y) → (↑x + ↑y) * f x y = y * (f x (x + y)))
: (answer = f 14 52)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q319
find_all (answer : ℝ) iff
  (f : ℕ → ℕ → ℝ)
  (h₀ : ∀ x, 0 < x → f x x = x)
  (h₁ : ∀ x y, (0 < x ∧ 0 < y) → f x y = f y x)
  (h₂ : ∀ x y, (0 < x ∧ 0 < y) → (↑x + ↑y) * f x y = y * (f x (x + y)))
: (answer = f 14 52)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
