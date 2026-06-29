import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What digit must be placed in the blank to make the four-digit integer $20\_7$ a multiple of 11?

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q240.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q240
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : n ≤ 9)
  (h₁ : 11 ∣ 20 * 100 + 10 * n + 7)
: (answer = n)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q240
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : n ≤ 9)
  (h₁ : 11 ∣ 20 * 100 + 10 * n + 7)
: (answer = n)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
