import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
An audience of $450$ people is seated in an auditorium. Each row contains the same number of seats and each seat in the auditorium is occupied. With three fewer seats per row, and five extra rows, the same audience could still be seated, occupying all seats. How many rows does the auditorium have?

## Answer
25
-/

-- # Formal Answer
noncomputable abbrev Q50.gt := fun (answer : ℕ) => answer = (25 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q50
find (answer : ℕ) s.t.
  (rows : ℕ)
  (seats : ℕ)
  (h₀ : rows * seats = 450)
  (h₁ : (rows + 5) * (seats - 3) = 450)
: (answer = rows)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q50
find_all (answer : ℕ) iff
  (rows : ℕ)
  (seats : ℕ)
  (h₀ : rows * seats = 450)
  (h₁ : (rows + 5) * (seats - 3) = 450)
: (answer = rows)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
