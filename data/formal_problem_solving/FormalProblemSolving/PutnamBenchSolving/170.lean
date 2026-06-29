import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
Let $N_n$ denote the number of ordered $n$-tuples of positive integers $(a_1,a_2,\ldots,a_n)$ such that $1/a_1 + 1/a_2 +\ldots + 1/a_n=1$.  Determine whether $N_{10}$ is even or odd.

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q170.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q170
find (answer : Prop) s.t.
  (N : (n : ℕ+) → Set (Fin n → ℕ+))
  (hN : N = fun (n : ℕ+) => {t : Fin n → ℕ+ | (∀ i j : Fin n, i < j → t i <= t j) ∧ (∑ i : Fin n, (1 : ℝ)/(t i) = 1) })
: (Odd (N 10).ncard ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q170
find_all (answer : Prop) iff
  (N : (n : ℕ+) → Set (Fin n → ℕ+))
  (hN : N = fun (n : ℕ+) => {t : Fin n → ℕ+ | (∀ i j : Fin n, i < j → t i <= t j) ∧ (∑ i : Fin n, (1 : ℝ)/(t i) = 1) })
: (Odd (N 10).ncard ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
