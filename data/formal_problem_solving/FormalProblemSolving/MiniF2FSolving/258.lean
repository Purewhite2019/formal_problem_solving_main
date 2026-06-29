import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given $2^a = 32$ and $a^b = 125$ find $b^a$.

## Answer
243
-/

-- # Formal Answer
noncomputable abbrev Q258.gt := fun (answer : ℝ) => answer = (243 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q258
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h₀ : (2 : ℝ)^a = 32)
  (h₁ : a^b = 125)
: (answer = b^a)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q258
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h₀ : (2 : ℝ)^a = 32)
  (h₁ : a^b = 125)
: (answer = b^a)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
