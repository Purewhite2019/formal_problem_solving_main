import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Evaluate $\lceil (3.6)^2 \rceil - ( \lceil 3.6 \rceil ) ^2$.

## Answer
-3
-/

-- # Formal Answer
noncomputable abbrev Q124.gt := fun (answer : ℝ) => answer = (-3 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q124
find (answer : ℝ) s.t.
: (answer = Int.ceil ((3.6:ℝ)^2) - (Int.ceil (3.6:ℝ))^2)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q124
find_all (answer : ℝ) iff
: (answer = Int.ceil ((3.6:ℝ)^2) - (Int.ceil (3.6:ℝ))^2)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
