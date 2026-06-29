import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
When a number is divided by 5, the remainder is 3. What is the remainder when twice the number is divided by 5?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q372.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q372
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : n % 5 = 3)
: (answer = (2 * n) % 5)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q372
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : n % 5 = 3)
: (answer = (2 * n) % 5)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
