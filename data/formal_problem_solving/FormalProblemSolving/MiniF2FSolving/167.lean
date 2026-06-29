import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Both roots of the quadratic equation $x^2 - 63x + k = 0$ are prime numbers. The number of possible values of $k$ is

$\text{(A)}\ 0 \qquad \text{(B)}\ 1 \qquad \text{(C)}\ 2 \qquad \text{(D)}\ 4 \qquad \text{(E) more than 4}$

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q167.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q167
find (answer : ℕ) s.t.
: (answer = Nat.card {k : ℕ | ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p + q = 63 ∧ p * q = k})
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q167
find_all (answer : ℕ) iff
: (answer = Nat.card {k : ℕ | ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p + q = 63 ∧ p * q = k})
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
