import FormalProblemSolving.Basic
open Polynomial

/-
## Problem
Simplify $(-k + 4) + (-2 + 3k)$.

## Answer
2k+2
-/

-- # Formal Answer
noncomputable abbrev Q4.gt := fun (answer : Polynomial ℝ) => answer = (((2 : ℝ[X]) * (Polynomial.X : ℝ[X]) : ℝ[X]) + (2 : ℝ[X]) : ℝ[X])

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q4
find (answer : Polynomial ℝ) s.t.
: (answer = (-X + 4) + (-2 + 3*X))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q4
find_all (answer : Polynomial ℝ) iff
: (answer = (-X + 4) + (-2 + 3*X))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
