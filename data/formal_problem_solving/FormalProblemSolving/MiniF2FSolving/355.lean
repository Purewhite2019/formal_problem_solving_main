import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose that $P = 2^m$ and $Q = 3^n$. Which of the following is equal to $12^{mn}$ for every pair of integers $(m,n)$?

$\textbf{(A)}\ P^2Q \qquad \textbf{(B)}\ P^nQ^m \qquad \textbf{(C)}\ P^nQ^{2m} \qquad \textbf{(D)}\ P^{2m}Q^n \qquad \textbf{(E)}\ P^{2n}Q^m$

## Answer
P^{2n} Q^m
-/

-- # Formal Answer
noncomputable abbrev Q355.gt := fun (m : ℝ) (n : ℝ) (answer : ℝ) => answer = ((12 : ℝ) ^ (m * n : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q355
  (m : ℝ)
  (n : ℝ)
find (answer : ℝ) s.t.
  (P : ℝ)
  (Q : ℝ)
  (h₀ : P = 2 ^ m)
  (h₁ : Q = 3 ^ n)
: (answer = P^(2 * n) * Q^m)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q355
  (m : ℝ)
  (n : ℝ)
find_all (answer : ℝ) iff
  (P : ℝ)
  (Q : ℝ)
  (h₀ : P = 2 ^ m)
  (h₁ : Q = 3 ^ n)
: (answer = P^(2 * n) * Q^m)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
