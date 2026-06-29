import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the greatest integer less than $(\sqrt{7} + \sqrt{5})^6.$  (Do not use a calculator!)

## Answer
13535
-/

-- # Formal Answer
noncomputable abbrev Q325.gt := fun (answer : ℤ) => answer = (13535 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q325
find (answer : ℤ) s.t.
: (answer < (Real.sqrt 7 + Real.sqrt 5)^6 ∧ ∀ n : ℤ, n < (Real.sqrt 7 + Real.sqrt 5)^6 → n ≤ answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q325
find_all (answer : ℤ) iff
: (answer < (Real.sqrt 7 + Real.sqrt 5)^6 ∧ ∀ n : ℤ, n < (Real.sqrt 7 + Real.sqrt 5)^6 → n ≤ answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
