import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the sum of the four positive factors of the positive integer value of $\sqrt{196}$?

## Answer
24
-/

-- # Formal Answer
noncomputable abbrev Q57.gt := fun (answer : ℕ) => answer = (24 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q57
find (answer : ℕ) s.t.
  (S : Finset ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ n ∣ (Nat.sqrt 196) ∧ n > 0)
: (answer = ∑ k in S, k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q57
find_all (answer : ℕ) iff
  (S : Finset ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ n ∣ (Nat.sqrt 196) ∧ n > 0)
: (answer = ∑ k in S, k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
