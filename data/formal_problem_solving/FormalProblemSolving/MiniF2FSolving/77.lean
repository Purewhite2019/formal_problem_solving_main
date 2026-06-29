import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $a * b = a^b + b^a$, for all positive integer values of $a$ and $b$, then what is the value of $2 * 6$?

## Answer
100
-/

-- # Formal Answer
noncomputable abbrev Q77.gt := fun (answer : ℕ) => answer = (100 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q77
find (answer : ℕ) s.t.
  (s : ℕ → ℕ → ℕ)
  (h₀ : ∀ a b, 0 < a ∧ 0 < b → s a b = a^b + b^a)
: (answer = s 2 6)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q77
find_all (answer : ℕ) iff
  (s : ℕ → ℕ → ℕ)
  (h₀ : ∀ a b, 0 < a ∧ 0 < b → s a b = a^b + b^a)
: (answer = s 2 6)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
