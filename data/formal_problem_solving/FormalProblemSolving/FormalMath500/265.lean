import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The binary number $10101001110_{2}$ is equal to what number in base eight?

## Answer
2516_8
-/

-- # Formal Answer
noncomputable abbrev Q265.gt := fun (answer : List ℕ) => answer = ((6 : ℕ) :: ((1 : ℕ) :: ((5 : ℕ) :: ((2 : ℕ) :: ([] : List ℕ) : List ℕ) : List ℕ) : List ℕ) : List ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q265
find (answer : List ℕ) s.t.
: (answer = Nat.digits 8 (Nat.ofDigits 2 [0,1,1,1,0,0,1,0,1,0,1]))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q265
find_all (answer : List ℕ) iff
: (answer = Nat.digits 8 (Nat.ofDigits 2 [0,1,1,1,0,0,1,0,1,0,1]))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
