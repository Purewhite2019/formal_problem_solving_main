import FormalProblemSolving.Basic

/-
## Problem
Let $\mathcal{P}$ be the set of vectors defined by $\mathcal{P}=\left\{\left.\begin{pmatrix} a \\ b \end{pmatrix}\right| 0 \leq a \leq 2, 0 \leq b \leq 100,\text{ and }a,b \in \mathbb{Z}\right\}$. Find all $\mathbf{v} \in \mathcal{P}$ such that the set $\mathcal{P} \setminus \{\mathbf{v}\}$ obtained by omitting vector $\mathbf{v}$ from $\mathcal{P}$ can be partitioned into two sets of equal size and equal sum.

## Answer
{v : Mathlib.Vector ℤ 2 | ∃ b : ℤ, 0 ≤ b ∧ b ≤ 100 ∧ Even b ∧ v.toList = [1, b]}
-/

-- # Formal Answer
noncomputable abbrev Q279.gt := fun (v : Fin 2 → ℤ) (answer : Set (Fin 2 → ℤ)) => answer =
  ({v : Fin (2 : ℕ) → ℤ |
      ∃ (b : ℤ),
        (0 : ℤ) ≤ b ∧
          b ≤ (100 : ℤ) ∧
            Even b ∧
              v =
                (Matrix.vecCons (1 : ℤ) (Matrix.vecCons b (![] : Fin (0 : ℕ) → ℤ) : Fin (Nat.succ (0 : ℕ) : ℕ) → ℤ) :
                  Fin ((Nat.succ (0 : ℕ) : ℕ).succ : ℕ) → ℤ)} :
    Set (Fin (2 : ℕ) → ℤ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q279
  (v : Fin 2 → ℤ)
find (answer : Set (Fin 2 → ℤ)) s.t.
  (P : Finset (Fin 2 → ℤ))
  (Pvdiff : Finset (Fin 2 → ℤ))
  (hP : P = {v' : Fin 2 → ℤ | 0 ≤ v' 0 ∧ v' 0 ≤ 2 ∧ 0 ≤ v' 1 ∧ v' 1 ≤ 100})
  (hPvdiff : Pvdiff = P \ ({v} : Finset (Fin 2 → ℤ)))
: ((v ∈ P ∧ (∃ Q R : Finset (Fin 2 → ℤ),
    (Q ∪ R = Pvdiff) ∧ (Q ∩ R = ∅) ∧ (Q.card = R.card) ∧
    (∑ q in Q, q 0 = ∑ r in R, r 0) ∧ (∑ q in Q, q 1 = ∑ r in R, r 1)))
  ↔ v ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q279
  (v : Fin 2 → ℤ)
find_all (answer : Set (Fin 2 → ℤ)) iff
  (P : Finset (Fin 2 → ℤ))
  (Pvdiff : Finset (Fin 2 → ℤ))
  (hP : P = {v' : Fin 2 → ℤ | 0 ≤ v' 0 ∧ v' 0 ≤ 2 ∧ 0 ≤ v' 1 ∧ v' 1 ≤ 100})
  (hPvdiff : Pvdiff = P \ ({v} : Finset (Fin 2 → ℤ)))
: ((v ∈ P ∧ (∃ Q R : Finset (Fin 2 → ℤ),
    (Q ∪ R = Pvdiff) ∧ (Q ∩ R = ∅) ∧ (Q.card = R.card) ∧
    (∑ q in Q, q 0 = ∑ r in R, r 0) ∧ (∑ q in Q, q 1 = ∑ r in R, r 1)))
  ↔ v ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
