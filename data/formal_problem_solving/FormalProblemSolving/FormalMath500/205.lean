import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Evaluate $\lceil{\sqrt{20}}\rceil^2$.

## Answer
25
-/

-- # Formal Answer
noncomputable abbrev Q205.gt := fun (answer : ℝ) => answer = (25 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q205
find (answer : ℝ) s.t.
: (answer = (Nat.ceil (Real.sqrt 20))^2)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q205
find_all (answer : ℝ) iff
: (answer = (Nat.ceil (Real.sqrt 20))^2)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
