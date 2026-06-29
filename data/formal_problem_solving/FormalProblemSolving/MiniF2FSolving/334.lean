import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The first three terms of an arithmetic sequence are $2x - 3$, $5x - 11$, and $3x + 1$ respectively. The $n$th term of the sequence is $2009$. What is $n$?

$\textbf{(A)}\ 255 \qquad \textbf{(B)}\ 502 \qquad \textbf{(C)}\ 1004 \qquad \textbf{(D)}\ 1506 \qquad \textbf{(E)}\ 8037$

## Answer
502
-/

-- # Formal Answer
noncomputable abbrev Q334.gt := fun (answer : ℕ) => answer = (502 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q334
find (answer : ℕ) s.t.
  (x : ℤ)
  (a : ℕ → ℤ)
  (h_arith : ∀ m : ℕ, a (m + 3) - a (m + 2) = a (m + 2) - a (m + 1))
  (h1 : a 1 = 2 * x - 3)
  (h2 : a 2 = 5 * x - 11)
  (h3 : a 3 = 3 * x + 1)
: (1 ≤ answer ∧ a answer = 2009)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q334
find_all (answer : ℕ) iff
  (x : ℤ)
  (a : ℕ → ℤ)
  (h_arith : ∀ m : ℕ, a (m + 3) - a (m + 2) = a (m + 2) - a (m + 1))
  (h1 : a 1 = 2 * x - 3)
  (h2 : a 2 = 5 * x - 11)
  (h3 : a 3 = 3 * x + 1)
: (1 ≤ answer ∧ a answer = 2009)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
