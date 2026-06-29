import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Together, Amy and Betty have 20 apples. Amy has three times the number of apples that Betty has. How many more apples than Betty does Amy have?

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q163.gt := fun (answer : ℕ) => answer = (10 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q163
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (h₀ : 0 < a ∧ 0 < b)
  (h₁ : a + b = 20)
  (h₂ : a = 3 * b)
: (answer = a - b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q163
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (h₀ : 0 < a ∧ 0 < b)
  (h₁ : a + b = 20)
  (h₂ : a = 3 * b)
: (answer = a - b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
