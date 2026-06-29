import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
When Rachel divides her favorite number by 7, she gets a remainder of 5. What will the remainder be if she multiplies her favorite number by 5 and then divides by 7?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q56.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q56
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : n % 7 = 5)
: (answer = (5 * n) % 7)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q56
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : n % 7 = 5)
: (answer = (5 * n) % 7)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
