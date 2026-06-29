import FormalProblemSolving.Basic
open Filter Topology Nat

/-
## Problem
Consider a paper punch that can be centered at any point of the plane and that, when operated, removes from the plane precisely those points whose distance from the center is irrational. How many punches are needed to remove every point?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q132.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q132
find (answer : ℕ) s.t.
: (sInf {n : ℕ | ∃ S : Set (EuclideanSpace ℝ (Fin 2)), S.encard = n ∧ ∀ Q : EuclideanSpace ℝ (Fin 2), ∃ P ∈ S, Irrational (dist P Q)} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q132
find_all (answer : ℕ) iff
: (sInf {n : ℕ | ∃ S : Set (EuclideanSpace ℝ (Fin 2)), S.encard = n ∧ ∀ Q : EuclideanSpace ℝ (Fin 2), ∃ P ∈ S, Irrational (dist P Q)} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
