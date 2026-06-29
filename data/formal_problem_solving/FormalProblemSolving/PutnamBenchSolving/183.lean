import FormalProblemSolving.Basic
open Topology Filter

/-
## Problem
Let $A$ be a positive real number. What are the possible values of $\sum_{j=0}^\infty x_j^2$, given that $x_0,x_1,\ldots$ are positive numbers for which $\sum_{j=0}^\infty x_j=A$?

## Answer
the open interval $(0, A^2)$
-/

-- # Formal Answer
noncomputable abbrev Q183.gt := fun (A : ℝ) (answer : ℝ → Set ℝ) => answer = fun (A : ℝ) => (Set.Ioo (0 : ℝ) (A ^ (2 : ℕ) : ℝ) : Set ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q183
  (A : ℝ)
find (answer : ℝ → Set ℝ) s.t.
  (Apos : A > 0)
: ({S : ℝ |
    ∃ x : ℕ → ℝ,
      (∀ j : ℕ, x j > 0) ∧
      (∑' j : ℕ, x j) = A ∧
      (∑' j : ℕ, (x j) ^ 2) = S}
  = answer A)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q183
  (A : ℝ)
find_all (answer : ℝ → Set ℝ) iff
  (Apos : A > 0)
: ({S : ℝ |
    ∃ x : ℕ → ℝ,
      (∀ j : ℕ, x j > 0) ∧
      (∑' j : ℕ, x j) = A ∧
      (∑' j : ℕ, (x j) ^ 2) = S}
  = answer A)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
