import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The product of three consecutive positive integers is $8$ times their sum. What is the sum of their [[perfect square|squares]]?

$\mathrm{(A)}\ 50
\qquad\mathrm{(B)}\ 77
\qquad\mathrm{(C)}\ 110
\qquad\mathrm{(D)}\ 149
\qquad\mathrm{(E)}\ 194$

## Answer
77
-/

-- # Formal Answer
noncomputable abbrev Q259.gt := fun (answer : ℕ) => answer = (77 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q259
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : b = a + 1)
  (h₂ : c = b + 1)
  (h₃ : a * b * c = 8 * (a + b + c))
: (answer = a^2 + b^2 + c^2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q259
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : b = a + 1)
  (h₂ : c = b + 1)
  (h₃ : a * b * c = 8 * (a + b + c))
: (answer = a^2 + b^2 + c^2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
