import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $3a+2b=5$ and $a+b=2$, what is the ordered pair $(a,b)$ that satisfies both equations?

## Answer
(1, 1)
-/

-- # Formal Answer
noncomputable abbrev Q321.gt := fun (answer : ℝ × ℝ) => answer = (((1 : ℝ), (1 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q321
find (answer : ℝ × ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h₀ : 3 * a + 2 * b = 5)
  (h₁ : a + b = 2)
: (answer = (a, b))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q321
find_all (answer : ℝ × ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h₀ : 3 * a + 2 * b = 5)
  (h₁ : a + b = 2)
: (answer = (a, b))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
