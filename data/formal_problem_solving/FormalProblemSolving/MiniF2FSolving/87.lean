import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The graph of $y=ax^2 + bx + c$ is a parabola with vertical axis of symmetry.  The vertex of this parabola is $(2,3)$ and the parabola contains the point $(4,4)$.  Find the value of $y$ when $x=6$.

## Answer
7
-/

-- # Formal Answer
noncomputable abbrev Q87.gt := fun (answer : ℝ) => answer = (7 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q87
find (answer : ℝ) s.t.
  (a : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = a * (x - 2)^2 + 3)
  (h₁ : f 4 = 4)
: (answer = f 6)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q87
find_all (answer : ℝ) iff
  (a : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = a * (x - 2)^2 + 3)
  (h₁ : f 4 = 4)
: (answer = f 6)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
