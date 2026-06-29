import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the [[reciprocal]] of $\frac{1}{2}+\frac{2}{3}$?

$\mathrm{(A)}\ \frac{6}{7}\qquad\mathrm{(B)}\ \frac{7}{6}\qquad\mathrm{(C)}\ \frac{5}{3}\qquad\mathrm{(D)}\ 3\qquad\mathrm{(E)}\ \frac{7}{2}$

## Answer
6/7
-/

-- # Formal Answer
noncomputable abbrev Q54.gt := fun (answer : ℚ) => answer = (6 / 7 : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q54
find (answer : ℚ) s.t.
: (answer = 1 / (1 / 2 + 2 / 3))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q54
find_all (answer : ℚ) iff
: (answer = 1 / (1 / 2 + 2 / 3))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
