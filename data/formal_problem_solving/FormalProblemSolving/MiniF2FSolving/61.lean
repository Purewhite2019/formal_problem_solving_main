import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Consider the sequence of numbers: $4,7,1,8,9,7,6,\dots$ For $n>2$, the $n$-th term of the sequence is the units digit of the sum of the two previous terms. Let $S_n$ denote the sum of the first $n$ terms of this sequence. The smallest value of $n$ for which $S_n>10,000$ is:

$
\text{(A) }1992
\qquad
\text{(B) }1999
\qquad
\text{(C) }2001
\qquad
\text{(D) }2002
\qquad
\text{(E) }2004
$

## Answer
(B) 1999
-/

-- # Formal Answer
noncomputable abbrev Q61.gt := fun (answer : ℕ) => answer = (1999 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q61
find (answer : ℕ) s.t.
  (u : ℕ → ℕ)
  (h₀ : u 0 = 4)
  (h₁ : u 1 = 7)
  (h₂ : ∀ n : ℕ, u (n + 2) = (u n + u (n + 1)) % 10)
  (S : ℕ → ℕ)
  (hS : ∀ n, S n = ∑ k in Finset.range n, u k)
: (IsLeast {n | S n > 10000} answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q61
find_all (answer : ℕ) iff
  (u : ℕ → ℕ)
  (h₀ : u 0 = 4)
  (h₁ : u 1 = 7)
  (h₂ : ∀ n : ℕ, u (n + 2) = (u n + u (n + 1)) % 10)
  (S : ℕ → ℕ)
  (hS : ∀ n, S n = ∑ k in Finset.range n, u k)
: (IsLeast {n | S n > 10000} answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
