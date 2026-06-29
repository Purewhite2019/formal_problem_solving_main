import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the value of $$
(3x-2)(4x+1)-(3x-2)4x+1
$$ when $x=4$?

## Answer
11
-/

-- # Formal Answer
noncomputable abbrev Q125.gt := fun (answer : ℝ) => answer = (11 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q125
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : x = 4)
: ((3 * x - 2) * (4 * x + 1) - (3 * x - 2) * 4 * x + 1 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q125
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : x = 4)
: ((3 * x - 2) * (4 * x + 1) - (3 * x - 2) * 4 * x + 1 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
