import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What positive two-digit integer is exactly twice the sum of its digits?

## Answer
18
-/

-- # Formal Answer
noncomputable abbrev Q92.gt := fun (answer : ℕ) => answer = (18 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q92
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (h₀ : 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9)
  (h₁ : 10 * a + b = 2 * (a + b))
: (answer = 10 * a + b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q92
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (h₀ : 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9)
  (h₁ : 10 * a + b = 2 * (a + b))
: (answer = 10 * a + b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
