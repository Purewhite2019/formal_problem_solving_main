import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many integers are in the solution set of $|x-2|\leq5.6$ ?

## Answer
11
-/

-- # Formal Answer
noncomputable abbrev Q272.gt := fun (answer : ℕ) => answer = (11 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q272
find (answer : ℕ) s.t.
  (S : Finset ℤ)
  (h₀ : ∀ (n : ℤ), n ∈ S ↔ abs (n - 2) ≤ 5 + 6 / 10)
: (answer = S.card)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q272
find_all (answer : ℕ) iff
  (S : Finset ℤ)
  (h₀ : ∀ (n : ℤ), n ∈ S ↔ abs (n - 2) ≤ 5 + 6 / 10)
: (answer = S.card)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
