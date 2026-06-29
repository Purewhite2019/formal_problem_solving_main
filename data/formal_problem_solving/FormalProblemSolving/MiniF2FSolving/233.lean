import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the distance between the two intersections of $y=x^2$ and $x+y=1$?

## Answer
\sqrt{10}
-/

-- # Formal Answer
noncomputable abbrev Q233.gt := fun (answer : ℝ) => answer = (√(10 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q233
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (h₀ : b = a^2)
  (h₁ : a + b = 1)
  (h₂ : d = c^2)
  (h₃ : c + d = 1)
  (h₄ : a ≠ c)
: (answer = Real.sqrt ((a - c)^2 + (b - d)^2))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q233
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (h₀ : b = a^2)
  (h₁ : a + b = 1)
  (h₂ : d = c^2)
  (h₃ : c + d = 1)
  (h₄ : a ≠ c)
: (answer = Real.sqrt ((a - c)^2 + (b - d)^2))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
