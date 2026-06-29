import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The midpoint of the line segment between $(x,y)$ and $(-9,1)$ is $(3,-5)$. Find $(x,y)$.

## Answer
(15, -11)
-/

-- # Formal Answer
noncomputable abbrev Q52.gt := fun (answer : ℝ × ℝ) => answer = (((15 : ℝ), (-11 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q52
find (answer : ℝ × ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : (3, -5) = ((x + (-9)) / 2, (y + 1) / 2))
: (answer = (x, y))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q52
find_all (answer : ℝ × ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : (3, -5) = ((x + (-9)) / 2, (y + 1) / 2))
: (answer = (x, y))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
