import FormalProblemSolving.Basic

/-
## Problem
Two different prime numbers between $4$ and $18$ are chosen. When their sum is subtracted from their product, which of the following numbers could be obtained?

$\textbf{(A)}\ 22 \qquad\textbf{(B)}\ 60 \qquad\textbf{(C)}\ 119 \qquad\textbf{(D)}\ 180 \qquad\textbf{(E)}\ 231$

## Answer
119
-/

-- # Formal Answer
noncomputable abbrev Q318.gt := fun (answer : ℕ) => answer = (119 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q318
find (answer : ℕ) s.t.
: answer ∈ ({22, 60, 119, 180, 231} : Set ℕ) ∧
  ∃ (p : ℕ) (q : ℕ),
    Nat.Prime p ∧ Nat.Prime q ∧
    (4 : ℕ) < p ∧ p < (18 : ℕ) ∧ (4 : ℕ) < q ∧ q < (18 : ℕ) ∧
    p ≠ q ∧ ((p * q : ℕ) - (p + q : ℕ) : ℕ) = answer
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q318
find_all (answer : ℕ) iff
: answer ∈ ({22, 60, 119, 180, 231} : Set ℕ) ∧
  ∃ (p : ℕ) (q : ℕ),
    Nat.Prime p ∧ Nat.Prime q ∧
    (4 : ℕ) < p ∧ p < (18 : ℕ) ∧ (4 : ℕ) < q ∧ q < (18 : ℕ) ∧
    p ≠ q ∧ ((p * q : ℕ) - (p + q : ℕ) : ℕ) = answer
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
