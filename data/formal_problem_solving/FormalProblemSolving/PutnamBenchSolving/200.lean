import FormalProblemSolving.Basic
open Nat Set

/-
## Problem
Find all positive integers $n,k_1,\dots,k_n$ such that $k_1+\cdots+k_n=5n-4$ and $\frac{1}{k_1}+\cdots+\frac{1}{k_n}=1$.

## Answer
{(n, k) : ℕ × (ℕ → ℤ) | (n = 1 ∧ k 0 = 1) ∨ (n = 3 ∧ (k '' {0, 1, 2} = {2, 3, 6})) ∨ (n = 4 ∧ (∀ i : Fin 4, k i = 4))}
-/

-- # Formal Answer
noncomputable abbrev Q200.gt := fun (answer : Set (ℕ × (ℕ → ℤ))) => answer =
  ({((n, k) : ℕ × (ℕ → ℤ)) : ℕ × (ℕ → ℤ) |
      n = (1 : ℕ) ∧ (k (0 : ℕ) : ℤ) = (1 : ℤ) ∨
        n = (3 : ℕ) ∧
            (k '' (Insert.insert (0 : ℕ) (Insert.insert (1 : ℕ) ({(2 : ℕ)} : Set ℕ) : Set ℕ) : Set ℕ) : Set ℤ) =
              (Insert.insert (2 : ℤ) (Insert.insert (3 : ℤ) ({(6 : ℤ)} : Set ℤ) : Set ℤ) : Set ℤ) ∨
          n = (4 : ℕ) ∧ ∀ (i : Fin (4 : ℕ)), (k (↑i : ℕ) : ℤ) = (4 : ℤ)} :
    Set (ℕ × (ℕ → ℤ)))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q200
find (answer : Set (ℕ × (ℕ → ℤ))) s.t.
: ({((n : ℕ), (k : ℕ → ℤ)) | (n > 0) ∧ (∀ i ∈ Finset.range n, k i > 0) ∧ (∑ i in Finset.range n, k i = 5 * n - 4) ∧ (∑ i : Finset.range n, (1 : ℝ) / (k i) = 1)} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q200
find_all (answer : Set (ℕ × (ℕ → ℤ))) iff
: ({((n : ℕ), (k : ℕ → ℤ)) | (n > 0) ∧ (∀ i ∈ Finset.range n, k i > 0) ∧ (∑ i in Finset.range n, k i = 5 * n - 4) ∧ (∑ i : Finset.range n, (1 : ℝ) / (k i) = 1)} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
