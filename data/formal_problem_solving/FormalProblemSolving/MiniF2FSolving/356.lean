import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sum of 5 consecutive even integers is 4 less than the sum of the first 8 consecutive odd counting numbers. What is the smallest of the even integers?

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q356.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q356
find (answer : ℕ) s.t.
  (a : ℕ)
  (h₀ : Even a)
  (h₁ : ∑ k in Finset.range 8, (2 * k + 1) - ∑ k in Finset.range 5, (a + 2 * k) = (4:ℤ))
: (answer = a)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q356
find_all (answer : ℕ) iff
  (a : ℕ)
  (h₀ : Even a)
  (h₁ : ∑ k in Finset.range 8, (2 * k + 1) - ∑ k in Finset.range 5, (a + 2 * k) = (4:ℤ))
: (answer = a)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
