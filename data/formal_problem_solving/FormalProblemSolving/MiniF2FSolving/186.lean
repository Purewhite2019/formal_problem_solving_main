import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sum of two nonzero real numbers is 4 times their product. What is the sum of the reciprocals of the two numbers?

$\textbf{(A)}\ 1\qquad\textbf{(B)}\ 2\qquad\textbf{(C)}\ 4\qquad\textbf{(D)}\ 8\qquad\textbf{(E)}\ 12$

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q186.gt := fun (answer : ℝ) => answer = (4 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q186
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : y ≠ 0)
  (h₂ : x + y = 4 * (x * y))
: (answer = 1 / x + 1 / y)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q186
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : y ≠ 0)
  (h₂ : x + y = 4 * (x * y))
: (answer = 1 / x + 1 / y)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
