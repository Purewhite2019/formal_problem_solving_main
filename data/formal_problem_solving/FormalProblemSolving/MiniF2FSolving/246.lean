import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The Fibonacci sequence is the sequence 1, 1, 2, 3, 5, $\ldots$ where each term is the sum of the previous two terms. What is the remainder when the $100^{\mathrm{th}}$ term of the sequence is divided by 4?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q246.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q246
find (answer : ℕ) s.t.
  (a : ℕ → ℕ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 1)
  (h₂ : ∀ n, a (n + 2) = a (n + 1) + a n)
: (answer = a 100 % 4)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q246
find_all (answer : ℕ) iff
  (a : ℕ → ℕ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 1)
  (h₂ : ∀ n, a (n + 2) = a (n + 1) + a n)
: (answer = a 100 % 4)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
