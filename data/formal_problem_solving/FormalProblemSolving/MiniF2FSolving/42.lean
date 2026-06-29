import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
At each basketball practice last week, Jenny made twice as many free throws as she made at the previous practice.  At her fifth practice she made 48 free throws.  How many free throws did she make at the first practice?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q42.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q42
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : 2 * (2 * (2 * (2 * x))) = 48)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q42
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : 2 * (2 * (2 * (2 * x))) = 48)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
