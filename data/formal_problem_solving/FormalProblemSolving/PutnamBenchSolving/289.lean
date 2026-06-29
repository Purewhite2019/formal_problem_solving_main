import FormalProblemSolving.Basic
open Topology Filter Set Matrix

/-
## Problem
Let \( \mathbb{Z}^n \) be the integer lattice in \( \mathbb{R}^n \). Two points in \( \mathbb{Z}^n \) are called neighbors if they differ by exactly 1 in one coordinate and are equal in all other coordinates. For which integers \( n \geq 1 \) does there exist a set of points \( S \subset \mathbb{Z}^n \) satisfying the following two conditions? \begin{enumerate} \item If \( p \) is in \( S \), then none of the neighbors of \( p \) is in \( S \). \item If \( p \in \mathbb{Z}^n \) is not in \( S \), then exactly one of the neighbors of \( p \) is in \( S \). \end{enumerate}

## Answer
the set of all integers n ≥ 1
-/

-- # Formal Answer
noncomputable abbrev Q289.gt := fun (answer : Set ℕ) => answer = (Set.Ici (1 : ℕ) : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q289
find (answer : Set ℕ) s.t.
  (n : ℕ)
  (neighbors : (Fin n → ℤ) → (Fin n → ℤ) → Prop)
  (neighbors_def : ∀ p q, neighbors p q ↔ (∃ i : Fin n, abs (p i - q i) = 1 ∧ ∀ j ≠ i, p j = q j))
: ((1 ≤ n ∧ ∃ S : Set (Fin n → ℤ),
    (∀ p ∈ S, ∀ q, neighbors p q → q ∉ S) ∧ (∀ p ∉ S, {q ∈ S | neighbors p q}.encard = 1)) ↔ n ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q289
find_all (answer : Set ℕ) iff
  (n : ℕ)
  (neighbors : (Fin n → ℤ) → (Fin n → ℤ) → Prop)
  (neighbors_def : ∀ p q, neighbors p q ↔ (∃ i : Fin n, abs (p i - q i) = 1 ∧ ∀ j ≠ i, p j = q j))
: ((1 ≤ n ∧ ∃ S : Set (Fin n → ℤ),
    (∀ p ∈ S, ∀ q, neighbors p q → q ∉ S) ∧ (∀ p ∉ S, {q ∈ S | neighbors p q}.encard = 1)) ↔ n ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
