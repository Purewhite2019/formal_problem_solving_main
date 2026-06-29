import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
In an arithmetic sequence, the 7th term is 30, and the 11th term is 60. What is the 21st term of this sequence?

## Answer
135
-/

-- # Formal Answer
noncomputable abbrev Q323.gt := fun (answer : ℝ) => answer = (135 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q323
find (answer : ℝ) s.t.
  (a : ℝ)
  (d : ℝ)
  (h₀ : a + 6 * d = 30)
  (h₁ : a + 10 * d = 60)
: (answer = a + 20 * d)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q323
find_all (answer : ℝ) iff
  (a : ℝ)
  (d : ℝ)
  (h₀ : a + 6 * d = 30)
  (h₁ : a + 10 * d = 60)
: (answer = a + 20 * d)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
