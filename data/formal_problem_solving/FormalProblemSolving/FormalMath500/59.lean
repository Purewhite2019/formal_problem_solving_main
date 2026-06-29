import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The arithmetic mean of 7, 2, $x$ and 10 is 9. What is the value of $x$?

## Answer
17
-/

-- # Formal Answer
noncomputable abbrev Q59.gt := fun (answer : ℝ) => answer = (17 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q59
find (answer : ℝ) s.t.
  (x : ℝ)
  (h_mean : (7 + 2 + x + 10) / 4 = 9)
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q59
find_all (answer : ℝ) iff
  (x : ℝ)
  (h_mean : (7 + 2 + x + 10) / 4 = 9)
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
