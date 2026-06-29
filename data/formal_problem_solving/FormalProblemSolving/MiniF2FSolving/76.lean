import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose that $\left|x+y\right|+\left|x-y\right|=2$. What is the maximum possible value of $x^2-6x+y^2$? $
\textbf{(A)}\ 5 \qquad
\textbf{(B)}\ 6 \qquad
\textbf{(C)}\ 7 \qquad
\textbf{(D)}\ 8 \qquad
\textbf{(E)}\ 9 $

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q76.gt := fun (answer : ℝ) => answer = (8 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q76
find (answer : ℝ) s.t.
: (∀ (x y : ℝ),
    ((|x + y|) + (|x - y|)) = 2 →
      ((x ^ 2 - (6 * x)) + (y ^ 2)) ≤ answer) ∧
  ∃ (x : ℝ) (y : ℝ),
    ((|(x + y)|) + (|(x - y)|)) = 2 ∧
      ((x ^ 2 - (6 * x)) + (y ^ 2)) = answer
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q76
find_all (answer : ℝ) iff
: (∀ (x y : ℝ),
    ((|x + y|) + (|x - y|)) = 2 →
      ((x ^ 2 - (6 * x)) + (y ^ 2)) ≤ answer) ∧
  ∃ (x : ℝ) (y : ℝ),
    ((|(x + y)|) + (|(x - y)|)) = 2 ∧
      ((x ^ 2 - (6 * x)) + (y ^ 2)) = answer
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
