import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The graphs of $y=x^4$ and $y=5x^2-6$ intersect at four points with $x$-coordinates $\pm \sqrt{m}$ and $\pm \sqrt{n}$, where $m > n$. What is $m-n$?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q218.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q218
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀t, f t = t^4)
  (h₁ : ∀t, g t = 5 * t^2 - 6)
  (h₂ : f x = g x)
  (h₃ : f y = g y)
  (h₄ : x^2 < y^2)
: (answer = y^2 - x^2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q218
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀t, f t = t^4)
  (h₁ : ∀t, g t = 5 * t^2 - 6)
  (h₂ : f x = g x)
  (h₃ : f y = g y)
  (h₄ : x^2 < y^2)
: (answer = y^2 - x^2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
