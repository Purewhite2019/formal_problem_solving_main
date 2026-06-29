import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The square root of $t$ is greater than $2$ and less than $3.5$. How many integer values of $t$ satisfy this condition?

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q28.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q28
find (answer : ℕ) s.t.
  (S : Finset ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ Real.sqrt n < 3.5 ∧ 2 < Real.sqrt n)
: (answer = S.card)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q28
find_all (answer : ℕ) iff
  (S : Finset ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ Real.sqrt n < 3.5 ∧ 2 < Real.sqrt n)
: (answer = S.card)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
