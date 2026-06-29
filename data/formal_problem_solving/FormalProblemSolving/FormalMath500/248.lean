import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Express $555_{10}$ in base $5$.

## Answer
4210_{5}
-/

-- # Formal Answer
noncomputable abbrev Q248.gt := fun (answer : List ℕ) => answer = ((4 : ℕ) :: ((2 : ℕ) :: ((1 : ℕ) :: ((0 : ℕ) :: ([] : List ℕ) : List ℕ) : List ℕ) : List ℕ) : List ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q248
find (answer : List ℕ) s.t.
: (answer = (Nat.digits 5 555).reverse)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q248
find_all (answer : List ℕ) iff
: (answer = (Nat.digits 5 555).reverse)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
