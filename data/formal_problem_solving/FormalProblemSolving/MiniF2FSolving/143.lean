import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For what value of $x$ does $10^{x}\cdot 100^{2x}=1000^{5}$?

$\textbf{(A)}\ 1 \qquad\textbf{(B)}\ 2\qquad\textbf{(C)}\ 3\qquad\textbf{(D)}\ 4\qquad\textbf{(E)}\ 5$

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q143.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q143
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : (10:ℝ)^x * 100^(2 * x) = 1000^5)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q143
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : (10:ℝ)^x * 100^(2 * x) = 1000^5)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
