import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Two non-zero [[real number]]s, $a$ and $b,$ satisfy $ab = a - b$. Which of the following is a possible value of $\frac {a}{b} + \frac {b}{a} - ab$?

$\textbf{(A)} \ - 2 \qquad \textbf{(B)} \ \frac { -1 }{2} \qquad \textbf{(C)} \ \frac {1}{3} \qquad \textbf{(D)} \ \frac {1}{2} \qquad \textbf{(E)} \ 2$

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q107.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q107
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁ : a * b = a - b)
: (answer = a / b + b / a - a * b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q107
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁ : a * b = a - b)
: (answer = a / b + b / a - a * b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
