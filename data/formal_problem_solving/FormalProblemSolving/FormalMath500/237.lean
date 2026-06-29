import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute $58_9 - 18_9.$ Express your answer in base $9.$

## Answer
40_9
-/

-- # Formal Answer
noncomputable abbrev Q237.gt := fun (answer : List ℕ) => answer = ((4 : ℕ) :: ((0 : ℕ) :: ([] : List ℕ) : List ℕ) : List ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q237
find (answer : List ℕ) s.t.
: (Nat.ofDigits 9 (answer.reverse) = (Nat.ofDigits 9 [8,5]) - (Nat.ofDigits 9 [8,1]))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q237
find_all (answer : List ℕ) iff
: (Nat.ofDigits 9 (answer.reverse) = (Nat.ofDigits 9 [8,5]) - (Nat.ofDigits 9 [8,1]))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
