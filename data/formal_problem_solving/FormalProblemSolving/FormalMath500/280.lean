import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the product of $6_8 \cdot 7_8.$ Express your answer in base $8.$

## Answer
52_8
-/

-- # Formal Answer
noncomputable abbrev Q280.gt := fun (answer : List ℕ) => answer = ((5 : ℕ) :: ((2 : ℕ) :: ([] : List ℕ) : List ℕ) : List ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q280
find (answer : List ℕ) s.t.
: (answer = ((Nat.digits 8 ((Nat.ofDigits 8 [6])*(Nat.ofDigits 8 [7]))).reverse))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q280
find_all (answer : List ℕ) iff
: (answer = ((Nat.digits 8 ((Nat.ofDigits 8 [6])*(Nat.ofDigits 8 [7]))).reverse))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
