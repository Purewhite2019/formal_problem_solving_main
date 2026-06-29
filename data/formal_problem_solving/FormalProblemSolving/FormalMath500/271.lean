import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Determine the modulo 4 remainder of the following sum: $$ 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12. $$

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q271.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q271
find (answer : ℕ) s.t.
  (sum : ℕ)
  (h_sum : sum = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)
: (sum % 4 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q271
find_all (answer : ℕ) iff
  (sum : ℕ)
  (h_sum : sum = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)
: (sum % 4 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
