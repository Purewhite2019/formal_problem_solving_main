import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the remainder when $$33818^2 + 33819^2 + 33820^2 + 33821^2 + 33822^2$$is divided by 17.

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q164.gt := fun (answer : ℕ) => answer = (0 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q164
find (answer : ℕ) s.t.
: (answer = (33818^2 + 33819^2 + 33820^2 + 33821^2 + 33822^2) % 17)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q164
find_all (answer : ℕ) iff
: (answer = (33818^2 + 33819^2 + 33820^2 + 33821^2 + 33822^2) % 17)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
