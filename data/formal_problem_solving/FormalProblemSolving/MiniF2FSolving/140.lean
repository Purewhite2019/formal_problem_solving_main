import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify $(a-1)(a+1)(a+2) - (a-2)(a+1).$

## Answer
a^3 + a^2
-/

-- # Formal Answer
noncomputable abbrev Q140.gt := fun (a : ℝ) (answer : ℝ) => answer = ((a ^ (3 : ℕ) : ℝ) + (a ^ (2 : ℕ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q140
  (a : ℝ)
find (answer : ℝ) s.t.
: (answer = (a - 1) * (a + 1) * (a + 2) - (a - 2) * (a + 1))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q140
  (a : ℝ)
find_all (answer : ℝ) iff
: (answer = (a - 1) * (a + 1) * (a + 2) - (a - 2) * (a + 1))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
