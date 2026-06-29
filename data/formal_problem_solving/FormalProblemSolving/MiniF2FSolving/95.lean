import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given that $x$ and $y$ are distinct nonzero real numbers such that $x+\tfrac{2}{x} = y + \tfrac{2}{y}$, what is $xy$?

$ \textbf{(A)}\ \frac{1}{4}\qquad\textbf{(B)}\ \frac{1}{2}\qquad\textbf{(C)}\ 1\qquad\textbf{(D)}\ 2\qquad\textbf{(E)}\ 4\qquad $

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q95.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q95
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : y ≠ 0)
  (h₂ : x ≠ y)
  (h₃ : x + 2 / x = y + 2 / y)
: (answer = x * y)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q95
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : y ≠ 0)
  (h₂ : x ≠ y)
  (h₃ : x + 2 / x = y + 2 / y)
: (answer = x * y)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
