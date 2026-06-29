import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
One dimension of a cube is increased by $1$, another is decreased by $1$, and the third is left unchanged. The volume of the new rectangular solid is $5$ less than that of the cube. What was the volume of the cube?

$\textbf{(A)}\ 8 \qquad \textbf{(B)}\ 27 \qquad \textbf{(C)}\ 64 \qquad \textbf{(D)}\ 125 \qquad \textbf{(E)}\ 216$

## Answer
125
-/

-- # Formal Answer
noncomputable abbrev Q26.gt := fun (answer : ℝ) => answer = (125 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q26
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : x^3 - (x + 1) * (x - 1) * x = 5)
: (answer = x^3)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q26
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : x^3 - (x + 1) * (x - 1) * x = 5)
: (answer = x^3)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
