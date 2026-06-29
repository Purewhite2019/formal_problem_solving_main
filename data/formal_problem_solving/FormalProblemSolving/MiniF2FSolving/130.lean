import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many distinct, non-equilateral triangles with a perimeter of 60 units have integer side lengths $a$, $b$, and $c$ such that $a$, $b$, $c$ is an arithmetic sequence?

## Answer
9
-/

-- # Formal Answer
noncomputable abbrev Q130.gt := fun (answer : ℕ) => answer = (9 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q130
find (answer : ℕ) s.t.
  (S : Set (ℕ × ℕ × ℕ))
  (hS : S = { (a, b, c) | 0 < a ∧ 0 < b ∧ 0 < c ∧ a ≤ b ∧ b ≤ c ∧ a + b + c = 60 ∧ 2 * b = a + c ∧ a + b > c ∧ a < c })
: (answer = Nat.card S)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q130
find_all (answer : ℕ) iff
  (S : Set (ℕ × ℕ × ℕ))
  (hS : S = { (a, b, c) | 0 < a ∧ 0 < b ∧ 0 < c ∧ a ≤ b ∧ b ≤ c ∧ a + b + c = 60 ∧ 2 * b = a + c ∧ a + b > c ∧ a < c })
: (answer = Nat.card S)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
