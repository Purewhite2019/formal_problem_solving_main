import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The square root of $t$ is greater than $2$ and less than $3.5$. How many integer values of $t$ satisfy this condition?

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q183.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q183
find (answer : ℕ) s.t.
  (S : Set ℤ)
  (hS : S = {t:ℤ | Real.sqrt t > 2 ∧ Real.sqrt t < 3.5})
: (S.encard = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q183
find_all (answer : ℕ) iff
  (S : Set ℤ)
  (hS : S = {t:ℤ | Real.sqrt t > 2 ∧ Real.sqrt t < 3.5})
: (S.encard = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
