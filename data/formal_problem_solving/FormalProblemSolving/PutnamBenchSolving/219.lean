import FormalProblemSolving.Basic
open Metric Filter Topology Set Nat

/-
## Problem
What is the largest possible radius of a circle contained in a $4$-dimensional hypercube of side length $1$?

## Answer
√2 / 2
-/

-- # Formal Answer
noncomputable abbrev Q219.gt := fun (answer : ℝ) => answer = ((√(2 : ℝ) : ℝ) / (2 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q219
find (answer : ℝ) s.t.
  (H : Set (EuclideanSpace ℝ (Fin 4)))
  (H_def : H = {P : Fin 4 → ℝ | ∀ i : Fin 4, |P i| ≤ 1 / 2})
  (contains : ℝ → Prop)
  (contains_def : ∀ r, contains r ↔
      ∃ᵉ (A : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 4))) (C ∈ A),
        Module.finrank ℝ A.direction = 2 ∧
        sphere C r ∩ A ⊆ H)
: (IsGreatest contains answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q219
find_all (answer : ℝ) iff
  (H : Set (EuclideanSpace ℝ (Fin 4)))
  (H_def : H = {P : Fin 4 → ℝ | ∀ i : Fin 4, |P i| ≤ 1 / 2})
  (contains : ℝ → Prop)
  (contains_def : ∀ r, contains r ↔
      ∃ᵉ (A : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 4))) (C ∈ A),
        Module.finrank ℝ A.direction = 2 ∧
        sphere C r ∩ A ⊆ H)
: (IsGreatest contains answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
