import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the value of  $(3x - 2)(4x + 1) - (3x - 2)4x + 1$ when $x=4$?

$\mathrm{(A)}\ 0
\qquad\mathrm{(B)}\ 1
\qquad\mathrm{(C)}\ 10
\qquad\mathrm{(D)}\ 11
\qquad\mathrm{(E)}\ 12$

## Answer
\mathrm{(D)}\ 11
-/

-- # Formal Answer
noncomputable abbrev Q330.gt := fun (answer : ℤ) => answer = (11 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q330
find (answer : ℤ) s.t.
  (x : ℤ)
  (h₀ : x = 4)
: (answer = (3 * x - 2) * (4 * x + 1) - (3 * x - 2) * (4 * x) + 1)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q330
find_all (answer : ℤ) iff
  (x : ℤ)
  (h₀ : x = 4)
: (answer = (3 * x - 2) * (4 * x + 1) - (3 * x - 2) * (4 * x) + 1)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
