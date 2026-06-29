import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A class of 30 students recently took a test.  If 20 students scored 80, 8 students scored 90, and 2 students scored 100, then what was the class average (the mean) on this test?

## Answer
84
-/

-- # Formal Answer
noncomputable abbrev Q55.gt := fun (answer : ℕ) => answer = (84 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q55
find (answer : ℕ) s.t.
  (total_students : ℕ)
  (students_80 : ℕ)
  (students_90 : ℕ)
  (students_100 : ℕ)
  (total_score : ℕ)
  (hs : total_students = 30 ∧ students_80 = 20 ∧ students_90 = 8 ∧ students_100 = 2)
  (ht : total_score = students_80 * 80 + students_90 * 90 + students_100 * 100)
: (total_score / total_students = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q55
find_all (answer : ℕ) iff
  (total_students : ℕ)
  (students_80 : ℕ)
  (students_90 : ℕ)
  (students_100 : ℕ)
  (total_score : ℕ)
  (hs : total_students = 30 ∧ students_80 = 20 ∧ students_90 = 8 ∧ students_100 = 2)
  (ht : total_score = students_80 * 80 + students_90 * 90 + students_100 * 100)
: (total_score / total_students = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
