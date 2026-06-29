import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find $ax^5 + by^5$ if the real numbers $a,b,x,$ and $y$ satisfy the equations
$\begin{align*}
ax + by &= 3, \\
ax^2 + by^2 &= 7, \\
ax^3 + by^3 &= 16, \\
ax^4 + by^4 &= 42.
\end{align*}$

## Answer
20
-/

-- # Formal Answer
noncomputable abbrev Q345.gt := fun (answer : ℝ) => answer = (20 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q345
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (x : ℝ)
  (y : ℝ)
  (h₀ : a * x + b * y = 3)
  (h₁ : a * x^2 + b * y^2 = 7)
  (h₂ : a * x^3 + b * y^3 = 16)
  (h₃ : a * x^4 + b * y^4 = 42)
: (answer = a * x^5 + b * y^5)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q345
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (x : ℝ)
  (y : ℝ)
  (h₀ : a * x + b * y = 3)
  (h₁ : a * x^2 + b * y^2 = 7)
  (h₂ : a * x^3 + b * y^3 = 16)
  (h₃ : a * x^4 + b * y^4 = 42)
: (answer = a * x^5 + b * y^5)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
