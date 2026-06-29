import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $321_{b}$ is equal to the base 10 integer 57, find $b$ given that $b>0$.

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q21.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q21
find (answer : ℕ) s.t.
  (b : ℕ)
  (h₀ : 0 < b)
  (h₁ : 3 * b^2 + 2 * b + 1 = 57)
: (answer = b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q21
find_all (answer : ℕ) iff
  (b : ℕ)
  (h₀ : 0 < b)
  (h₁ : 3 * b^2 + 2 * b + 1 = 57)
: (answer = b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
