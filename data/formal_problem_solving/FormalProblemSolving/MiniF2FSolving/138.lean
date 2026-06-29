import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the simplified numerical value of $\frac{a+11b}{a-b}$ if $\frac{4a+3b}{a-2b}=5$?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q138.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q138
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h₀ : a ≠ b)
  (h₁ : a ≠ 2 * b)
  (h₂ : (4 * a + 3 * b) / (a - 2 * b) = 5)
: (answer = (a + 11 * b) / (a - b))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q138
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h₀ : a ≠ b)
  (h₁ : a ≠ 2 * b)
  (h₂ : (4 * a + 3 * b) / (a - 2 * b) = 5)
: (answer = (a + 11 * b) / (a - b))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
