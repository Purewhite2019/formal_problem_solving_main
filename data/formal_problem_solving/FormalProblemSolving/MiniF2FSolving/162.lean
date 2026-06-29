import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The product of three consecutive integers is 720. What is the largest of these integers?

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q162.gt := fun (answer : ℕ) => answer = (10 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q162
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : (n - 1) * n * (n + 1) = 720)
: (answer = n + 1)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q162
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : (n - 1) * n * (n + 1) = 720)
: (answer = n + 1)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
