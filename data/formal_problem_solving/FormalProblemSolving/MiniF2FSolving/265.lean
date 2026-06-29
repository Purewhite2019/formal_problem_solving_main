import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If

\begin{align*}
3x+4y-12z&=10,\\
-2x-3y+9z&=-4,
\end{align*}

compute $x$.

## Answer
14
-/

-- # Formal Answer
noncomputable abbrev Q265.gt := fun (answer : ℝ) => answer = (14 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q265
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h₀ : 3 * x + 4 * y - 12 * z = 10)
  (h₁ : -2 * x - 3 * y + 9 * z = -4)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q265
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h₀ : 3 * x + 4 * y - 12 * z = 10)
  (h₁ : -2 * x - 3 * y + 9 * z = -4)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
