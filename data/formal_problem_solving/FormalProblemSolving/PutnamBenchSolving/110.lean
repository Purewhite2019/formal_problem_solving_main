import FormalProblemSolving.Basic

/-
## Problem
The sequence of digits $123456789101112131415161718192021 \dots$ is obtained by writing the positive integers in order. If the $10^n$-th digit in this sequence occurs in the part of the sequence in which the $m$-digit numbers are placed, define $f(n)$ to be $m$. For example, $f(2)=2$ because the $100$th digit enters the sequence in the placement of the two-digit integer $55$. Find, with proof, $f(1987)$.

## Answer
1984
-/

-- # Formal Answer
noncomputable abbrev Q110.gt := fun (answer : ℕ) => answer = (1984 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q110
find (answer : ℕ) s.t.
  (seqind : ℕ → ℕ)
  (seqsize : ℕ → ℕ)
  (f : ℕ → ℕ)
  (hseqind : seqind 1 = 1 ∧ ∀ i ≥ 2, seqind i = seqind (i - 1) + (Nat.digits 10 (i - 1)).length)
  (hseqsize : ∀ i ≥ 1, ∀ j : Fin ((Nat.digits 10 i).length), seqsize (seqind i + j) = (Nat.digits 10 i).length)
  (hf : ∀ n : ℕ, f n = seqsize (10 ^ n))
: (f 1987 = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q110
find_all (answer : ℕ) iff
  (seqind : ℕ → ℕ)
  (seqsize : ℕ → ℕ)
  (f : ℕ → ℕ)
  (hseqind : seqind 1 = 1 ∧ ∀ i ≥ 2, seqind i = seqind (i - 1) + (Nat.digits 10 (i - 1)).length)
  (hseqsize : ∀ i ≥ 1, ∀ j : Fin ((Nat.digits 10 i).length), seqsize (seqind i + j) = (Nat.digits 10 i).length)
  (hf : ∀ n : ℕ, f n = seqsize (10 ^ n))
: (f 1987 = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
