import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A line $\ell$ passes through the points $B(7,-1)$ and $C(-1,7)$.  The equation of this line can be written in the form $y=mx+b$; compute $m+b$.

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q369.gt := fun (answer : ℝ) => answer = (5 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q369
find (answer : ℝ) s.t.
  (m : ℝ)
  (b : ℝ)
  (h₀ : m * 7 + b = -1)
  (h₁ : m * (-1) + b = 7)
: (answer = m + b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q369
find_all (answer : ℝ) iff
  (m : ℝ)
  (b : ℝ)
  (h₀ : m * 7 + b = -1)
  (h₁ : m * (-1) + b = 7)
: (answer = m + b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
