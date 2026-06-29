import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many integers are in the solution set of $|x-2|\leq5.6$ ?

## Answer
11
-/

-- # Formal Answer
noncomputable abbrev Q111.gt := fun (answer : ℕ) => answer = (11 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q111
find (answer : ℕ) s.t.
  (S : Set ℤ)
  (hS : S = {x : ℤ | abs (x - 2) ≤  (28 : ℚ) / (5 : ℚ) })
: (S.encard = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q111
find_all (answer : ℕ) iff
  (S : Set ℤ)
  (hS : S = {x : ℤ | abs (x - 2) ≤  (28 : ℚ) / (5 : ℚ) })
: (S.encard = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
