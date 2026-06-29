import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $z$ be a complex number such that $|z| = 1.$  Find the maximum value of
\[|1 + z| + |1 - z + z^2|.\]

## Answer
\frac{13}{4}
-/

-- # Formal Answer
noncomputable abbrev Q318.gt := fun (answer : ℝ) => answer = ((↑(13 : ℚ) : ℝ) / (4 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q318
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {x : ℝ | ∃ z : ℂ, (Complex.abs z) = 1 ∧ x = (Complex.abs (1+z)) + (Complex.abs (1-z+z^2))})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q318
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {x : ℝ | ∃ z : ℂ, (Complex.abs z) = 1 ∧ x = (Complex.abs (1+z)) + (Complex.abs (1-z+z^2))})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
