import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many integers are in the solution of the inequality $|x + 4|< 9$?

## Answer
17
-/

-- # Formal Answer
noncomputable abbrev Q91.gt := fun (answer : ℕ) => answer = (17 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q91
find (answer : ℕ) s.t.
  (s : Finset ℤ)
  (f : ℤ → ℤ)
  (h₀ : ∀ x, f x = abs (x + 4))
  (h₁ : ∀ x, x ∈ s ↔ f x < 9)
: (answer = s.card)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q91
find_all (answer : ℕ) iff
  (s : Finset ℤ)
  (f : ℤ → ℤ)
  (h₀ : ∀ x, f x = abs (x + 4))
  (h₁ : ∀ x, x ∈ s ↔ f x < 9)
: (answer = s.card)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
