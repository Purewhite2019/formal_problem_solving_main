import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Factor the expression $ab+5b+2a+10$.

## Answer
(a+5)(b+2)
-/

-- # Formal Answer
noncomputable abbrev Q376.gt := fun (a : ℝ) (b : ℝ) (answer : ℝ) => answer = ((a + (5 : ℝ) : ℝ) * (b + (2 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q376
  (a : ℝ)
  (b : ℝ)
find (answer : ℝ) s.t.
: (answer = a * b + 5 * b + 2 * a + 10)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q376
  (a : ℝ)
  (b : ℝ)
find_all (answer : ℝ) iff
: (answer = a * b + 5 * b + 2 * a + 10)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
