import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The parabolas defined by the equations $y=x^2+4x+6$ and $y=\frac{1}{2}x^2+x+6$ intersect at points $(a,b)$ and $(c,d)$, where $c\ge a$. What is $c-a$?

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q155.gt := fun (answer : ℝ) => answer = (6 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q155
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (h₀ : b = a^2 + 4 * a + 6)
  (h₁ : b = 1 / 2 * a^2 + a + 6)
  (h₂ : d = c^2 + 4 * c + 6)
  (h₃ : d = 1 / 2 * c^2 + c + 6)
  (h₄ : a < c)
: (answer = c - a)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q155
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (h₀ : b = a^2 + 4 * a + 6)
  (h₁ : b = 1 / 2 * a^2 + a + 6)
  (h₂ : d = c^2 + 4 * c + 6)
  (h₃ : d = 1 / 2 * c^2 + c + 6)
  (h₄ : a < c)
: (answer = c - a)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
