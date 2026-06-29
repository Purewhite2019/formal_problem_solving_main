import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve for $x$: $\frac{x}2 + \frac{x}3 = 5$

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q14.gt := fun (answer : ℝ) => answer = (6 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q14
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : (x / 2) + (x / 3) = 5)
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q14
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : (x / 2) + (x / 3) = 5)
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
