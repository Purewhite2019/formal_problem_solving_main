import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A positive integer $X$ is 2 more than a multiple of 3. Its units digit is the same as the units digit of a number that is 4 more than a multiple of 5. What is the smallest possible value of $X$?

## Answer
14
-/

-- # Formal Answer
noncomputable abbrev Q212.gt := fun (answer : ℕ) => answer = (14 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q212
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (x : ℕ)
  (y : ℕ)
  (h₀ : x % 3 = 2)
  (h₁ : y % 5 = 4)
  (h₂ : x % 10 = y % 10)
  (hS : S = {x | x % 3 = 2 ∧ ∃ y, y % 5 = 4 ∧ x % 10 = y % 10})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q212
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (x : ℕ)
  (y : ℕ)
  (h₀ : x % 3 = 2)
  (h₁ : y % 5 = 4)
  (h₂ : x % 10 = y % 10)
  (hS : S = {x | x % 3 = 2 ∧ ∃ y, y % 5 = 4 ∧ x % 10 = y % 10})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
