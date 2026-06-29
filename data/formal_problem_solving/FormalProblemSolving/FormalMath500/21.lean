import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is $1^{(2^{235423523})}$?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q21.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q21
find (answer : ℕ) s.t.
: (1 ^ (2 ^ 235423523) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q21
find_all (answer : ℕ) iff
: (1 ^ (2 ^ 235423523) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
