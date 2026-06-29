import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The two-digit number $``B6,''$ where $B$ is the tens digit, is the square of a positive integer. How many distinct possible values are there for $B$?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q104.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q104
find (answer : ℕ) s.t.
  (b : ℕ)
  (h₀ : b < 10)
  (h₁ : ∃ n : ℕ, n^2 = 10 * b + 6)
  (S : Set ℕ)
  (hS : S = {b | b < 10 ∧ ∃ n : ℕ, n^2 = 10 * b + 6})
: (Nat.card S = answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q104
find_all (answer : ℕ) iff
  (b : ℕ)
  (h₀ : b < 10)
  (h₁ : ∃ n : ℕ, n^2 = 10 * b + 6)
  (S : Set ℕ)
  (hS : S = {b | b < 10 ∧ ∃ n : ℕ, n^2 = 10 * b + 6})
: (Nat.card S = answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
