import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Determine the remainder of 54 (mod 6).

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q267.gt := fun (answer : ℕ) => answer = (0 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q267
find (answer : ℕ) s.t.
: (54 % 6 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q267
find_all (answer : ℕ) iff
: (54 % 6 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
