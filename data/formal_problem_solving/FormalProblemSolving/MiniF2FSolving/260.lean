import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve  \[\frac{x-9}{x+1}=2\]for $x$.

## Answer
-11
-/

-- # Formal Answer
noncomputable abbrev Q260.gt := fun (answer : ℝ) => answer = (-11 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q260
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : x ≠ -1)
  (h₁ : (x - 9) / (x + 1) = 2)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q260
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : x ≠ -1)
  (h₁ : (x - 9) / (x + 1) = 2)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
