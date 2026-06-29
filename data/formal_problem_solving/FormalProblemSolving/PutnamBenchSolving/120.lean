import FormalProblemSolving.Basic
open Set Filter Topology

/-
## Problem
If a linear transformation $A$ on an $n$-dimensional vector space has $n+1$ eigenvectors such that any $n$ of them are linearly independent, does it follow that $A$ is a scalar multiple of the identity? Prove your answer.

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q120.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q120
find (answer : Prop) s.t.
: ((∀ (F V : Type*) (_ : Field F) (_ : AddCommGroup V) (_ : Module F V) (_ : FiniteDimensional F V) (n : ℕ) (A : Module.End F V) (evecs : Set V), (n = Module.finrank F V ∧ evecs ⊆ {v : V | ∃ f : F, A.HasEigenvector f v} ∧ evecs.encard = n + 1 ∧ (∀ sevecs : Fin n → V, (Set.range sevecs ⊆ evecs ∧ (Set.range sevecs).encard = n) → LinearIndependent F sevecs)) → (∃ c : F, A = c • LinearMap.id)) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q120
find_all (answer : Prop) iff
: ((∀ (F V : Type*) (_ : Field F) (_ : AddCommGroup V) (_ : Module F V) (_ : FiniteDimensional F V) (n : ℕ) (A : Module.End F V) (evecs : Set V), (n = Module.finrank F V ∧ evecs ⊆ {v : V | ∃ f : F, A.HasEigenvector f v} ∧ evecs.encard = n + 1 ∧ (∀ sevecs : Fin n → V, (Set.range sevecs ⊆ evecs ∧ (Set.range sevecs).encard = n) → LinearIndependent F sevecs)) → (∃ c : F, A = c • LinearMap.id)) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
