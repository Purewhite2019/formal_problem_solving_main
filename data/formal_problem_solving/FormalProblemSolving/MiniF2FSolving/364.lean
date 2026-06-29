import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $d(n)$ denote the number of positive integers that divide $n$, including $1$ and $n$. For example, $d(1)=1,d(2)=2,$ and $d(12)=6$. (This function is known as the divisor function.) Let$f(n)=\frac{d(n)}{\sqrt [3]n}.$There is a unique positive integer $N$ such that $f(N)>f(n)$ for all positive integers $n\ne N$. What is the sum of the digits of $N?$

$\textbf{(A) }5 \qquad \textbf{(B) }6 \qquad \textbf{(C) }7 \qquad \textbf{(D) }8\qquad \textbf{(E) }9$

## Answer
9
-/

-- # Formal Answer
noncomputable abbrev Q364.gt := fun (answer : ℕ) => answer = (9 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q364
find (answer : ℕ) s.t.
  (N : ℕ)
  (hN_pos : 0 < N)
  (h₁ : ∀ n, 0 < n → n ≠ N →
    ((Nat.divisors n).card : ℝ) / ((n : ℝ) ^ ((1 : ℝ)/3)) <
    ((Nat.divisors N).card : ℝ) / ((N : ℝ) ^ ((1 : ℝ)/3)))
: (answer = (Nat.digits 10 N).sum)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q364
find_all (answer : ℕ) iff
  (N : ℕ)
  (hN_pos : 0 < N)
  (h₁ : ∀ n, 0 < n → n ≠ N →
    ((Nat.divisors n).card : ℝ) / ((n : ℝ) ^ ((1 : ℝ)/3)) <
    ((Nat.divisors N).card : ℝ) / ((N : ℝ) ^ ((1 : ℝ)/3)))
: (answer = (Nat.digits 10 N).sum)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
