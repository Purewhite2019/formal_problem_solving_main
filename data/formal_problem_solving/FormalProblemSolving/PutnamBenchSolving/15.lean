import FormalProblemSolving.Basic
open Finset Polynomial Topology Filter Metric

/-
## Problem
Let $p$ be a prime number. Find the number of distinct $2 \times 2$ matrices $$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$$ such that $a, b, c, d \in \{0, 1, ..., p - 1\}$, $a + d \equiv 1 \pmod p$, and $ad - bc \equiv 0 \pmod p$.

## Answer
$p^2 + p$
-/

-- # Formal Answer
noncomputable abbrev Q15.gt := fun (p : ℕ) (answer : ℕ → ℕ) => answer = fun (p : ℕ) => ((p ^ (2 : ℕ) : ℕ) + p : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q15
  (p : ℕ)
find (answer : ℕ → ℕ) s.t.
  (hp : Nat.Prime p)
: ({M : Matrix (Fin 2) (Fin 2) (ZMod p) | M 0 0 + M 1 1 = 1 ∧ M 0 0 * M 1 1 - M 0 1 * M 1 0 = 0}.ncard = answer p)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q15
  (p : ℕ)
find_all (answer : ℕ → ℕ) iff
  (hp : Nat.Prime p)
: ({M : Matrix (Fin 2) (Fin 2) (ZMod p) | M 0 0 + M 1 1 = 1 ∧ M 0 0 * M 1 1 - M 0 1 * M 1 0 = 0}.ncard = answer p)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
