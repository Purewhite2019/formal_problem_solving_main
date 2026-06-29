import FormalProblemSolving.Basic
open Topology Filter Matrix

/-
## Problem
Let $S$ be the set of all ordered triples $(p,q,r)$ of prime numbers for which at least one rational number $x$ satisfies $px^2+qx+r=0$. Which primes appear in seven or more elements of $S$?

## Answer
{2, 5}
-/

-- # Formal Answer
noncomputable abbrev Q240.gt := fun (answer : Set ℕ) => answer = (Insert.insert (2 : ℕ) ({(5 : ℕ)} : Set ℕ) : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q240
find (answer : Set ℕ) s.t.
  (S : Set (Fin 3 → ℕ))
  (t : ℕ)
  (hS : S = {s : Fin 3 → ℕ | (s 0).Prime ∧ (s 1).Prime ∧ (s 2).Prime ∧ ∃ x : ℚ, (s 0) * x ^ 2 + (s 1) * x + (s 2) = 0})
: ((t.Prime ∧ ({s ∈ S | ∃ i : Fin 3, s i = t}.encard ≥ 7)) ↔ t ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q240
find_all (answer : Set ℕ) iff
  (S : Set (Fin 3 → ℕ))
  (t : ℕ)
  (hS : S = {s : Fin 3 → ℕ | (s 0).Prime ∧ (s 1).Prime ∧ (s 2).Prime ∧ ∃ x : ℚ, (s 0) * x ^ 2 + (s 1) * x + (s 2) = 0})
: ((t.Prime ∧ ({s ∈ S | ∃ i : Fin 3, s i = t}.encard ≥ 7)) ↔ t ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
