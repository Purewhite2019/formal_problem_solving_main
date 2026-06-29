import FormalProblemSolving.Basic

/-
## Problem
Let $S$ be the set of all $2 \times 2$ real matrices $M=\begin{pmatrix} a & b \\ c & d \end{pmatrix}$ whose entries $a,b,c,d$ (in that order) form an arithmetic progression. Find all matrices $M$ in $S$ for which there is some integer $k>1$ such that $M^k$ is also in $S$.

## Answer
the set of all matrices of the form α * I or β * A, where I is the identity matrix and A is a specific matrix
-/

-- # Formal Answer
noncomputable abbrev Q259.gt := fun (answer : Set (Matrix (Fin 2) (Fin 2) ℝ)) => answer =
  ({A : Matrix (Fin (2 : ℕ)) (Fin (2 : ℕ)) ℝ |
      (∃ (α : ℝ), ∀ (i j : Fin (2 : ℕ)), (A i j : ℝ) = (α * (1 : ℝ) : ℝ)) ∨
        ∃ (β : ℝ),
          (A (0 : Fin (2 : ℕ)) (0 : Fin (2 : ℕ)) : ℝ) = (β * (-3 : ℝ) : ℝ) ∧
            (A (0 : Fin (2 : ℕ)) (1 : Fin (2 : ℕ)) : ℝ) = (β * (-1 : ℝ) : ℝ) ∧
              (A (1 : Fin (2 : ℕ)) (0 : Fin (2 : ℕ)) : ℝ) = (β * (1 : ℝ) : ℝ) ∧
                (A (1 : Fin (2 : ℕ)) (1 : Fin (2 : ℕ)) : ℝ) = (β * (3 : ℝ) : ℝ)} :
    Set (Matrix (Fin (2 : ℕ)) (Fin (2 : ℕ)) ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q259
find (answer : Set (Matrix (Fin 2) (Fin 2) ℝ)) s.t.
  (M : Matrix (Fin 2) (Fin 2) ℝ)
  (S : Set (Matrix (Fin 2) (Fin 2) ℝ))
  (hS : S = {M' | (M' 0 1 - M' 0 0 = M' 1 0 - M' 0 1) ∧ (M' 1 0 - M' 0 1 = M' 1 1 - M' 1 0)})
: (M ∈ S ∧ (∃ k > 1, M ^ k ∈ S) ↔ M ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q259
find_all (answer : Set (Matrix (Fin 2) (Fin 2) ℝ)) iff
  (M : Matrix (Fin 2) (Fin 2) ℝ)
  (S : Set (Matrix (Fin 2) (Fin 2) ℝ))
  (hS : S = {M' | (M' 0 1 - M' 0 0 = M' 1 0 - M' 0 1) ∧ (M' 1 0 - M' 0 1 = M' 1 1 - M' 1 0)})
: (M ∈ S ∧ (∃ k > 1, M ^ k ∈ S) ↔ M ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
