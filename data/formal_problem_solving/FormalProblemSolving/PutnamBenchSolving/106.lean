import FormalProblemSolving.Basic
open Real Equiv

/-
## Problem
Inscribe a rectangle of base $b$ and height $h$ and an isosceles triangle of base $b$ (against a corresponding side of the rectangle and pointed in the other direction) in a circle of radius one. For what value of $h$ do the rectangle and triangle have the same area?

## Answer
$2/5$
-/

-- # Formal Answer
noncomputable abbrev Q106.gt := fun (answer : ℝ) => answer = (2 / 5 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q106
find (answer : ℝ) s.t.
  (b : ℝ)
  (h : ℝ)
  (hbh : b > 0 ∧ h > 0 ∧ b ^ 2 + h ^ 2 = 2 ^ 2)
  (areaeq : b * h = 0.5 * b * (1 - h / 2))
: (h = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q106
find_all (answer : ℝ) iff
  (b : ℝ)
  (h : ℝ)
  (hbh : b > 0 ∧ h > 0 ∧ b ^ 2 + h ^ 2 = 2 ^ 2)
  (areaeq : b * h = 0.5 * b * (1 - h / 2))
: (h = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
