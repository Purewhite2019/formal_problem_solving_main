import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
At which point do the lines $s=9-2t$ and $t=3s+1$ intersect? Give your answer as an ordered pair in the form $(s, t).$

## Answer
(1, 4)
-/

-- # Formal Answer
noncomputable abbrev Q195.gt := fun (answer : ℝ × ℝ) => answer = (((1 : ℝ), (4 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q195
find (answer : ℝ × ℝ) s.t.
  (s : ℝ)
  (t : ℝ)
  (h₀ : s = 9 - 2 * t)
  (h₁ : t = 3 * s + 1)
: (answer = (s, t))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q195
find_all (answer : ℝ × ℝ) iff
  (s : ℝ)
  (t : ℝ)
  (h₀ : s = 9 - 2 * t)
  (h₁ : t = 3 * s + 1)
: (answer = (s, t))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
