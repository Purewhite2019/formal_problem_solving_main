import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
On planet Larky, 7 ligs = 4 lags, and 9 lags = 20 lugs. How many ligs are equivalent to 80 lugs?

## Answer
63
-/

-- # Formal Answer
noncomputable abbrev Q206.gt := fun (answer : ℝ) => answer = (63 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q206
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : 9 * b = 20 * c)
  (h₂ : 7 * a = 4 * b)
: (answer * a = 80 * c)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q206
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : 9 * b = 20 * c)
  (h₂ : 7 * a = 4 * b)
: (answer * a = 80 * c)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
