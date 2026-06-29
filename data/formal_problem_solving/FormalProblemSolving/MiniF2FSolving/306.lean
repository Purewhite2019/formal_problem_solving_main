import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
In the year $2001$, the United States will host the [[International Mathematical Olympiad]].  Let $I,M,$ and $O$ be distinct [[positive integer]]s such that the product $I \cdot M \cdot O = 2001 $.  What is the largest possible value of the sum $I + M + O$?

$\textbf{(A)}\ 23 \qquad \textbf{(B)}\ 55 \qquad \textbf{(C)}\ 99 \qquad \textbf{(D)}\ 111 \qquad \textbf{(E)}\ 671$

## Answer
671
-/

-- # Formal Answer
noncomputable abbrev Q306.gt := fun (answer : ℕ) => answer = (671 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q306
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (i : ℕ)
  (m : ℕ)
  (o : ℕ)
  (h₀ : i ≠ m ∧ m ≠ o ∧ o ≠ i)
  (h₁ : i * m * o = 2001)
  (hS : S = {s | ∃ (i m o : ℕ), i ≠ m ∧ m ≠ o ∧ o ≠ i ∧ i * m * o = 2001 ∧ s = i + m + o})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q306
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (i : ℕ)
  (m : ℕ)
  (o : ℕ)
  (h₀ : i ≠ m ∧ m ≠ o ∧ o ≠ i)
  (h₁ : i * m * o = 2001)
  (hS : S = {s | ∃ (i m o : ℕ), i ≠ m ∧ m ≠ o ∧ o ≠ i ∧ i * m * o = 2001 ∧ s = i + m + o})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
