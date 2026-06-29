import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Cards are numbered from 1 to 100. One card is removed and the values on the other 99 are added. The resulting sum is a multiple of 77. What number was on the card that was removed?

## Answer
45
-/

-- # Formal Answer
noncomputable abbrev Q36.gt := fun (answer : ℕ) => answer = (45 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q36
find (answer : ℕ) s.t.
: (answer ∈ Finset.Icc 1 100) ∧ (77 ∣ ∑ k in (Finset.Icc 1 100).erase answer, k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q36
find_all (answer : ℕ) iff
: (answer ∈ Finset.Icc 1 100) ∧ (77 ∣ ∑ k in (Finset.Icc 1 100).erase answer, k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
