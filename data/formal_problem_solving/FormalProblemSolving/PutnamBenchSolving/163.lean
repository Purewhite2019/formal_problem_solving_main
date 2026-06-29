import FormalProblemSolving.Basic

/-
## Problem
Suppose that each of 20 students has made a choice of anywhere from 0 to 6 courses from a total of 6 courses offered. Prove or disprove: there are 5 students and 2 courses such that all 5 have chosen both courses or all 5 have chosen neither course.

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q163.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q163
find (answer : Prop) s.t.
  (choices : Fin 20 → Set (Fin 6))
: (answer ↔ (∃ (students : Finset (Fin 20)) (courses : Finset (Fin 6)),
                        students.card = 5 ∧
                        courses.card = 2 ∧
                        (↑courses ⊆ ⋂ s ∈ students, choices s ∨ ↑courses ⊆ ⋂ s ∈ students, (choices s)ᶜ)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q163
find_all (answer : Prop) iff
  (choices : Fin 20 → Set (Fin 6))
: (answer ↔ (∃ (students : Finset (Fin 20)) (courses : Finset (Fin 6)),
                        students.card = 5 ∧
                        courses.card = 2 ∧
                        (↑courses ⊆ ⋂ s ∈ students, choices s ∨ ↑courses ⊆ ⋂ s ∈ students, (choices s)ᶜ)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
