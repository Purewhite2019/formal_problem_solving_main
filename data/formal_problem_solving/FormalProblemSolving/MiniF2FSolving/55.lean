import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $p (x) = 2-x^2$ and $q(x) = \frac{6}{x}$, what is the value of $p (q(2))$?

## Answer
-7
-/

-- # Formal Answer
noncomputable abbrev Q55.gt := fun (answer : ℝ) => answer = (-7 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q55
find (answer : ℝ) s.t.
  (p : ℝ → ℝ)
  (q : ℝ → ℝ)
  (h₀ : ∀ x, p x = 2 - x^2)
  (h₁ : ∀ x, x ≠ 0 → q x = 6 / x)
: (answer = p (q 2))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q55
find_all (answer : ℝ) iff
  (p : ℝ → ℝ)
  (q : ℝ → ℝ)
  (h₀ : ∀ x, p x = 2 - x^2)
  (h₁ : ∀ x, x ≠ 0 → q x = 6 / x)
: (answer = p (q 2))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
