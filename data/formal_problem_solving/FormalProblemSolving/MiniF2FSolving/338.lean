import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The product of two consecutive positive even integers is 288. What is the greater of the two integers?

## Answer
18
-/

-- # Formal Answer
noncomputable abbrev Q338.gt := fun (answer : ℕ) => answer = (18 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q338
find (answer : ℕ) s.t.
  (m : ℕ)
  (n : ℕ)
  (h₀ : Even m)
  (h₁ : Even n)
  (h₂ : m - n = 2)
  (h₃ : m * n = 288)
: (answer = m)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q338
find_all (answer : ℕ) iff
  (m : ℕ)
  (n : ℕ)
  (h₀ : Even m)
  (h₁ : Even n)
  (h₂ : m - n = 2)
  (h₃ : m * n = 288)
: (answer = m)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
