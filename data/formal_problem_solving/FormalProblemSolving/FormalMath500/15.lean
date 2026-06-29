import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Each week, between 30 and 50 students show up for an archery class run by Betty and Wilma.  Usually the students break up into groups of equal size for target practice.  However, this week, Betty noticed that she could not break the students up into multiple groups of equal size.  Wilma noticed that if she and Betty both joined the students in practicing, they still could not break the archers up into groups of equal size.  How many students showed up to the archery class this week?

## Answer
41
-/

-- # Formal Answer
noncomputable abbrev Q15.gt := fun (answer : ℕ) => answer = (41 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q15
find (answer : ℕ) s.t.
  (students : ℕ)
  (h₀ : 30 ≤ students ∧ students ≤ 50)
  (h₁ : ¬∃ (k : ℕ), 1 < k ∧ k < students ∧ students % k = 0)
  (h₂ : ¬∃ (k : ℕ), 1 < k ∧ k < students + 2 ∧ (students + 2) % k = 0)
: (students = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q15
find_all (answer : ℕ) iff
  (students : ℕ)
  (h₀ : 30 ≤ students ∧ students ≤ 50)
  (h₁ : ¬∃ (k : ℕ), 1 < k ∧ k < students ∧ students % k = 0)
  (h₂ : ¬∃ (k : ℕ), 1 < k ∧ k < students + 2 ∧ (students + 2) % k = 0)
: (students = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
