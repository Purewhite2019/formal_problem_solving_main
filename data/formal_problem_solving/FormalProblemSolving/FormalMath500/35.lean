import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Thirty-five students attend a math club meeting. The number of girls at the meeting is a multiple of 13, and there are more girls than boys attending the meeting. How many boys are at the meeting?

## Answer
9
-/

-- # Formal Answer
noncomputable abbrev Q35.gt := fun (answer : ℕ) => answer = (9 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q35
find (answer : ℕ) s.t.
  (total_students : ℕ)
  (girls : ℕ)
  (boys : ℕ)
  (h₀ : total_students = 35)
  (h₁ : girls + boys = total_students)
  (h₂ : ∃ k : ℕ, girls = 13 * k)
  (h₃ : girls > boys)
: (boys = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q35
find_all (answer : ℕ) iff
  (total_students : ℕ)
  (girls : ℕ)
  (boys : ℕ)
  (h₀ : total_students = 35)
  (h₁ : girls + boys = total_students)
  (h₂ : ∃ k : ℕ, girls = 13 * k)
  (h₃ : girls > boys)
: (boys = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
