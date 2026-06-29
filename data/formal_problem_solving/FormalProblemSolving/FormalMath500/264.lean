import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many positive whole-number divisors does 196 have?

## Answer
9
-/

-- # Formal Answer
noncomputable abbrev Q264.gt := fun (answer : ℕ) => answer = (9 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q264
find (answer : ℕ) s.t.
: (answer = (Nat.divisors 196).card)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q264
find_all (answer : ℕ) iff
: (answer = (Nat.divisors 196).card)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
