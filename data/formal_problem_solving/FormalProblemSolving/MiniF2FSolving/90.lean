import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $P(n)$ and $S(n)$ denote the product and the sum, respectively, of the digits
of the integer $n$. For example, $P(23) = 6$ and $S(23) = 5$. Suppose $N$ is a
two-digit number such that $N = P(N)+S(N)$. What is the units digit of $N$?

$\text{(A)}\ 2\qquad \text{(B)}\ 3\qquad \text{(C)}\ 6\qquad \text{(D)}\ 8\qquad \text{(E)}\ 9$

## Answer
9
-/

-- # Formal Answer
noncomputable abbrev Q90.gt := fun (answer : ℕ) => answer = (9 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q90
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (n : ℕ)
  (h₀ : 1 ≤ a ∧ a ≤ 9)
  (h₁ : 0 ≤ b ∧ b ≤ 9)
  (h₂ : n = 10 * a + b)
  (h₃ : n = a * b + a + b)
: (answer = b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q90
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (n : ℕ)
  (h₀ : 1 ≤ a ∧ a ≤ 9)
  (h₁ : 0 ≤ b ∧ b ≤ 9)
  (h₂ : n = 10 * a + b)
  (h₃ : n = a * b + a + b)
: (answer = b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
