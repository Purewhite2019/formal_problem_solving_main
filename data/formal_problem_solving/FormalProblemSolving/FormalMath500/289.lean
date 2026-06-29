import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Express the quotient $413_5 \div 2_5$ in base 5.

## Answer
204_5
-/

-- # Formal Answer
noncomputable abbrev Q289.gt := fun (answer : List ℕ) => answer = ((2 : ℕ) :: ((0 : ℕ) :: ((4 : ℕ) :: ([] : List ℕ) : List ℕ) : List ℕ) : List ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q289
find (answer : List ℕ) s.t.
: (answer = List.reverse (Nat.digits 5 ((Nat.ofDigits 5 (List.reverse [4,1,3])) / (Nat.ofDigits 5 (List.reverse [2])))))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q289
find_all (answer : List ℕ) iff
: (answer = List.reverse (Nat.digits 5 ((Nat.ofDigits 5 (List.reverse [4,1,3])) / (Nat.ofDigits 5 (List.reverse [2])))))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
