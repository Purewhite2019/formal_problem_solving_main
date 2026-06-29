import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A figure skater is facing north when she begins to spin to her right. She spins 2250 degrees. Which direction (north, south, east or west) is she facing when she finishes her spin?

## Answer
\text{east}
-/

-- # Formal Answer
noncomputable abbrev Q29.gt := fun (answer : ℕ) => answer = (90 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q29
find (answer : ℕ) s.t.
: (answer = 2250 % 360)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q29
find_all (answer : ℕ) iff
: (answer = 2250 % 360)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
