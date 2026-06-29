import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the base six equivalent of $999_{10}$?

## Answer
4343_6
-/

-- # Formal Answer
noncomputable abbrev Q293.gt := fun (answer : List ℕ) => answer = ((4 : ℕ) :: ((3 : ℕ) :: ((4 : ℕ) :: ((3 : ℕ) :: ([] : List ℕ) : List ℕ) : List ℕ) : List ℕ) : List ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q293
find (answer : List ℕ) s.t.
: (answer = (Nat.digits 6 999).reverse)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q293
find_all (answer : List ℕ) iff
: (answer = (Nat.digits 6 999).reverse)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
