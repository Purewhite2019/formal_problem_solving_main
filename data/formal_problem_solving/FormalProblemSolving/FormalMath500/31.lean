import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
In a class of 50 students, 28 participate in MATHCOUNTS, 21 participate in science club, and 6 students participate in neither. How many students participate in both MATHCOUNTS and science club?

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q31.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q31
find (answer : ℕ) s.t.
  (total_students : ℕ)
  (mathcounts : ℕ)
  (science_club : ℕ)
  (neither : ℕ)
  (both : ℕ)
  (h_total : total_students = 50)
  (h_mathcounts : mathcounts = 28)
  (h_science_club : science_club = 21)
  (h_neither : neither = 6)
  (h_both : mathcounts + science_club - both + neither = total_students)
: (both = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q31
find_all (answer : ℕ) iff
  (total_students : ℕ)
  (mathcounts : ℕ)
  (science_club : ℕ)
  (neither : ℕ)
  (both : ℕ)
  (h_total : total_students = 50)
  (h_mathcounts : mathcounts = 28)
  (h_science_club : science_club = 21)
  (h_neither : neither = 6)
  (h_both : mathcounts + science_club - both + neither = total_students)
: (both = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
