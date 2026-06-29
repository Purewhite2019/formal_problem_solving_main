import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
When 39,500 is divided by an integer $n$, the quotient is 123 and the remainder is 17. Find $n$.

## Answer
321
-/

-- # Formal Answer
noncomputable abbrev Q106.gt := fun (answer : ℕ) => answer = (321 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q106
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : 123 * n + 17 = 39500)
: (answer = n)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q106
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : 123 * n + 17 = 39500)
: (answer = n)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
