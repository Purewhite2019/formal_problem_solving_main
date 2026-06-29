import FormalProblemSolving.Basic
open Topology Filter Matrix

/-
## Problem
For which positive integers $n$ is there an $n \times n$ matrix with integer entries such that every dot product of a row with itself is even, while every dot product of two different rows is odd?

## Answer
the set of odd positive integers
-/

-- # Formal Answer
noncomputable abbrev Q239.gt := fun (n : ℕ) (answer : Set ℕ) => answer = ({n : ℕ | Odd n} : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q239
  (n : ℕ)
find (answer : Set ℕ) s.t.
  (nmat : ℕ → Prop)
  (hnmat : ∀ n, nmat n ↔
      ∃ A : Matrix (Fin n) (Fin n) ℤ,
        (∀ r, Even ((A r) ⬝ᵥ (A r))) ∧
        Pairwise fun r1 r2 ↦ Odd ((A r1) ⬝ᵥ (A r2)))
  (npos : 0 < n)
: (nmat n ↔ n ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q239
  (n : ℕ)
find_all (answer : Set ℕ) iff
  (nmat : ℕ → Prop)
  (hnmat : ∀ n, nmat n ↔
      ∃ A : Matrix (Fin n) (Fin n) ℤ,
        (∀ r, Even ((A r) ⬝ᵥ (A r))) ∧
        Pairwise fun r1 r2 ↦ Odd ((A r1) ⬝ᵥ (A r2)))
  (npos : 0 < n)
: (nmat n ↔ n ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
