import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For how many positive integer values of $x$ is the sum $x^2+4x+4$ less than 20?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q71.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q71
find (answer : ℕ) s.t.
  (S : Finset ℕ)
  (h₀ : ∀ x, x ∈ S ↔ 0 < x ∧ x^2 + 4 * x + 4 < 20)
: (answer = S.card)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q71
find_all (answer : ℕ) iff
  (S : Finset ℕ)
  (h₀ : ∀ x, x ∈ S ↔ 0 < x ∧ x^2 + 4 * x + 4 < 20)
: (answer = S.card)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
