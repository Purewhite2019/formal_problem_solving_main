import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the integer that lies between $\sqrt[3]{-45}$ and $\sqrt[3]{-101}$.

## Answer
-4
-/

-- # Formal Answer
noncomputable abbrev Q159.gt := fun (answer : ℤ) => answer = (-4 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q159
find (answer : ℤ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : x ^ 3 = -45)
  (h₁ : y ^ 3 = -101)
: ((answer : ℝ) < x) ∧ (y < (answer : ℝ))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q159
find_all (answer : ℤ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : x ^ 3 = -45)
  (h₁ : y ^ 3 = -101)
: ((answer : ℝ) < x) ∧ (y < (answer : ℝ))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
