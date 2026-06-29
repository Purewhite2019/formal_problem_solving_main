import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The product of two positive whole numbers is 2005. If neither number is 1, what is the sum of the two numbers?

## Answer
406
-/

-- # Formal Answer
noncomputable abbrev Q358.gt := fun (answer : ℕ) => answer = (406 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q358
find (answer : ℕ) s.t.
  (m : ℕ)
  (n : ℕ)
  (h₀ : 1 < m)
  (h₁ : 1 < n)
  (h₂ : m * n = 2005)
: (answer = m + n)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q358
find_all (answer : ℕ) iff
  (m : ℕ)
  (n : ℕ)
  (h₀ : 1 < m)
  (h₁ : 1 < n)
  (h₂ : m * n = 2005)
: (answer = m + n)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
