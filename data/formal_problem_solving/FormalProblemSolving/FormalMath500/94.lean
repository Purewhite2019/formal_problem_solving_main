import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let
\[f(x) = (\arccos x)^2 + (\arcsin x)^2.\]Find the range of $f(x).$  All functions are in radians.

## Answer
\left[ \frac{\pi^2}{8}, \frac{5 \pi^2}{4} \right]
-/

-- # Formal Answer
noncomputable abbrev Q94.gt := fun (answer : Set ℝ) => answer =
  ({y : ℝ | (((π : ℝ) ^ (2 : ℕ) : ℝ) / (8 : ℝ) : ℝ) ≤ y ∧ y ≤ (((5 : ℝ) * ((π : ℝ) ^ (2 : ℕ) : ℝ) : ℝ) / (4 : ℝ) : ℝ)} :
    Set ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q94
find (answer : Set ℝ) s.t.
: (answer = {y : ℝ | ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ y = (Real.arccos x)^2 + (Real.arcsin x)^2})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q94
find_all (answer : Set ℝ) iff
: (answer = {y : ℝ | ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ y = (Real.arccos x)^2 + (Real.arcsin x)^2})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
