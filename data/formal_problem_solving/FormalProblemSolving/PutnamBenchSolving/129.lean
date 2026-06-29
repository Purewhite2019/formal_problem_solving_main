import FormalProblemSolving.Basic
open Nat Filter Topology Set

/-
## Problem
Can a countably infinite set have an uncountable collection of non-empty subsets such that the intersection of any two of them is finite?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q129.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q129
find (answer : Prop) s.t.
: (answer ↔
    (∃ S : Type,
      Countable S ∧ Infinite S ∧
      ∃ C : Set (Set S),
        ¬Countable C ∧
        (∀ R ∈ C, R ≠ ∅) ∧
        (∀ A ∈ C, ∀ B ∈ C, A ≠ B → (A ∩ B).Finite)
    ))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q129
find_all (answer : Prop) iff
: (answer ↔
    (∃ S : Type,
      Countable S ∧ Infinite S ∧
      ∃ C : Set (Set S),
        ¬Countable C ∧
        (∀ R ∈ C, R ≠ ∅) ∧
        (∀ A ∈ C, ∀ B ∈ C, A ≠ B → (A ∩ B).Finite)
    ))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
