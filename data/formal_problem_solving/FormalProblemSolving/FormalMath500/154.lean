import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The 7th and 8th grades have enrollments of 520 and 650 respectively. The two grades have a total of 18 representatives on the Student Council. How many representatives should the 8th grade have if there is to be a fair representation of the two grades?

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q154.gt := fun (answer : ℕ) => answer = (10 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q154
find (answer : ℕ) s.t.
  (seventh_grade : ℕ)
  (enrollment_eighth_grade : ℕ)
  (total_representatives : ℕ)
  (h₀ : seventh_grade = 520)
  (h₁ : enrollment_eighth_grade = 650)
  (h₂ : total_representatives = 18)
: (((1:ℚ) * (enrollment_eighth_grade * total_representatives)) / (seventh_grade + enrollment_eighth_grade) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q154
find_all (answer : ℕ) iff
  (seventh_grade : ℕ)
  (enrollment_eighth_grade : ℕ)
  (total_representatives : ℕ)
  (h₀ : seventh_grade = 520)
  (h₁ : enrollment_eighth_grade = 650)
  (h₂ : total_representatives = 18)
: (((1:ℚ) * (enrollment_eighth_grade * total_representatives)) / (seventh_grade + enrollment_eighth_grade) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
