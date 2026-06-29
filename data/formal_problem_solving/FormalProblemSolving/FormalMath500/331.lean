import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the sum of all complex solutions of the equation
\[\frac{1}{x^2 - 1} + \frac{2}{x^2 - 2} + \frac{3}{x^2 - 3} + \frac{4}{x^2 - 4} = 2010x - 4.\]

## Answer
\frac{2}{1005}
-/

-- # Formal Answer
noncomputable abbrev Q331.gt := fun (answer : ℂ) => answer = (2 / 1005 : ℂ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q331
find (answer : ℂ) s.t.
  (SumSet : Finset ℂ)
  (h_sum : (SumSet : Set ℂ) = {x : ℂ | x^2 ≠ (1:ℂ) ∧ x^2 ≠ (2:ℂ) ∧ x^2 ≠ (3:ℂ) ∧ x^2 ≠ (4:ℂ) ∧
    (1:ℂ) / (x^2 - 1) + (2:ℂ) / (x^2 - 2) + (3:ℂ) / (x^2 - 3) + (4:ℂ) / (x^2 - 4) = 2010 * x - 4})
: (answer = ∑ x in SumSet, x)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q331
find_all (answer : ℂ) iff
  (SumSet : Finset ℂ)
  (h_sum : (SumSet : Set ℂ) = {x : ℂ | x^2 ≠ (1:ℂ) ∧ x^2 ≠ (2:ℂ) ∧ x^2 ≠ (3:ℂ) ∧ x^2 ≠ (4:ℂ) ∧
    (1:ℂ) / (x^2 - 1) + (2:ℂ) / (x^2 - 2) + (3:ℂ) / (x^2 - 3) + (4:ℂ) / (x^2 - 4) = 2010 * x - 4})
: (answer = ∑ x in SumSet, x)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
