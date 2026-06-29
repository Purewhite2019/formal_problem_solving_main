import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $x$, $y$, and $z$ are positive real numbers satisfying: \begin{align*}
\log x - \log y &= a, \\
\log y - \log z &= 15, \text{ and} \\
\log z - \log x &= -7, \\
\end{align*}where $a$ is a real number, what is $a$?

## Answer
-8
-/

-- # Formal Answer
noncomputable abbrev Q88.gt := fun (answer : ℝ) => answer = (-8 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q88
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (a : ℝ)
  (h₀ : 0 < x ∧ 0 < y ∧ 0 < z)
  (h₁ : Real.log x - Real.log y = a)
  (h₂ : Real.log y - Real.log z = 15)
  (h₃ : Real.log z - Real.log x = -7)
: (answer = a)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q88
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (a : ℝ)
  (h₀ : 0 < x ∧ 0 < y ∧ 0 < z)
  (h₁ : Real.log x - Real.log y = a)
  (h₂ : Real.log y - Real.log z = 15)
  (h₃ : Real.log z - Real.log x = -7)
: (answer = a)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
