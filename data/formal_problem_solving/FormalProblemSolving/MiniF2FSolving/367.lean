import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Assume that $x_1,x_2,\ldots,x_7$ are real numbers such that
$\begin{align*}
x_1 + 4x_2 + 9x_3 + 16x_4 + 25x_5 + 36x_6 + 49x_7 &= 1, \\
4x_1 + 9x_2 + 16x_3 + 25x_4 + 36x_5 + 49x_6 + 64x_7 &= 12, \\
9x_1 + 16x_2 + 25x_3 + 36x_4 + 49x_5 + 64x_6 + 81x_7 &= 123.
\end{align*}$
Find the value of $16x_1+25x_2+36x_3+49x_4+64x_5+81x_6+100x_7$.

## Answer
334
-/

-- # Formal Answer
noncomputable abbrev Q367.gt := fun (answer : ℝ) => answer = (334 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q367
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (e : ℝ)
  (f : ℝ)
  (g : ℝ)
  (h₀ : a + 4 * b + 9 * c + 16 * d + 25 * e + 36 * f + 49 * g = 1)
  (h₁ : 4 * a + 9 * b + 16 * c + 25 * d + 36 * e + 49 * f + 64 * g = 12)
  (h₂ : 9 * a + 16 * b + 25 * c + 36 * d + 49 * e + 64 * f + 81 * g = 123)
: (answer = 16 * a + 25 * b + 36 * c + 49 * d + 64 * e + 81 * f + 100 * g)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q367
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (e : ℝ)
  (f : ℝ)
  (g : ℝ)
  (h₀ : a + 4 * b + 9 * c + 16 * d + 25 * e + 36 * f + 49 * g = 1)
  (h₁ : 4 * a + 9 * b + 16 * c + 25 * d + 36 * e + 49 * f + 64 * g = 12)
  (h₂ : 9 * a + 16 * b + 25 * c + 36 * d + 49 * e + 64 * f + 81 * g = 123)
: (answer = 16 * a + 25 * b + 36 * c + 49 * d + 64 * e + 81 * f + 100 * g)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
