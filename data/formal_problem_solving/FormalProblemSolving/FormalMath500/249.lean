import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the units digit of $18^6.$

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q249.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q249
find (answer : ℕ) s.t.
: (answer = (18^6) % 10)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q249
find_all (answer : ℕ) iff
: (answer = (18^6) % 10)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
