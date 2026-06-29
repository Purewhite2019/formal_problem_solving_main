import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Zach has three bags and a bunch of pencils to be placed into the bags. He is told to place the greatest number of pencils possible into each of the three bags while also keeping the number of pencils in each bag the same. What is the greatest number of pencils he could have left over?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q240.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q240
find (answer : ℕ) s.t.
: ((∀ (pencils : ℕ), pencils % 3 ≤ answer) ∧ (∃ (pencils : ℕ), pencils % 3 = answer))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q240
find_all (answer : ℕ) iff
: ((∀ (pencils : ℕ), pencils % 3 ≤ answer) ∧ (∃ (pencils : ℕ), pencils % 3 = answer))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
