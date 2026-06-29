import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the difference between the sum of the first $2003$ even counting numbers and the sum of the first $2003$ odd counting numbers?

$ \mathrm{(A) \ } 0\qquad \mathrm{(B) \ } 1\qquad \mathrm{(C) \ } 2\qquad \mathrm{(D) \ } 2003\qquad \mathrm{(E) \ } 4006 $

## Answer
2003
-/

-- # Formal Answer
noncomputable abbrev Q139.gt := fun (answer : ℕ) => answer = (2003 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q139
find (answer : ℕ) s.t.
  (u : ℕ → ℕ)
  (v : ℕ → ℕ)
  (h₀ : ∀ n, u n = 2 * n + 2)
  (h₁ : ∀ n, v n = 2 * n + 1)
: (answer = (∑ k in Finset.range (2003), u k) - (∑ k in Finset.range (2003), v k))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q139
find_all (answer : ℕ) iff
  (u : ℕ → ℕ)
  (v : ℕ → ℕ)
  (h₀ : ∀ n, u n = 2 * n + 2)
  (h₁ : ∀ n, v n = 2 * n + 1)
: (answer = (∑ k in Finset.range (2003), u k) - (∑ k in Finset.range (2003), v k))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
