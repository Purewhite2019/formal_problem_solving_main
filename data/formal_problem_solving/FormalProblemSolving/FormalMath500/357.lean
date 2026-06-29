import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the real roots of
\[\frac{( x+ 1)(x - 3)}{5(x + 2)(x - 4)} + \frac{(x + 3)(x - 5)}{9(x + 4)(x - 6)} - \frac{2(x + 5)(x - 7)}{13(x + 6)(x - 8)} = \frac{92}{585}.\]Enter the real roots, separated by commas.

## Answer
1 \pm \sqrt{19}
-/

-- # Formal Answer
noncomputable abbrev Q357.gt := fun (answer : Set ℝ) => answer = (Insert.insert ((1 : ℝ) + (√(19 : ℝ) : ℝ) : ℝ) ({((1 : ℝ) - (√(19 : ℝ) : ℝ) : ℝ)} : Set ℝ) : Set ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q357
find (answer : Set ℝ) s.t.
: (answer = {x : ℝ | x ≠ -2 ∧ x ≠ 4 ∧ x ≠ -4 ∧ x ≠ 6 ∧ x ≠ -6 ∧ x ≠ 8 ∧
    (x + 1) * (x - 3) / (5 * (x + 2) * (x - 4)) + (x + 3) * (x - 5) / (9 * (x + 4) * (x - 6)) - 2 * (x + 5) * (x - 7) / (13 * (x + 6) * (x - 8)) = 92 / 585})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q357
find_all (answer : Set ℝ) iff
: (answer = {x : ℝ | x ≠ -2 ∧ x ≠ 4 ∧ x ≠ -4 ∧ x ≠ 6 ∧ x ≠ -6 ∧ x ≠ 8 ∧
    (x + 1) * (x - 3) / (5 * (x + 2) * (x - 4)) + (x + 3) * (x - 5) / (9 * (x + 4) * (x - 6)) - 2 * (x + 5) * (x - 7) / (13 * (x + 6) * (x - 8)) = 92 / 585})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
