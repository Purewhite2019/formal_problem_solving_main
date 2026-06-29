import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Two different positive numbers $a$ and $b$ each differ from their reciprocals by $1$. What is $a+b$?

$
\text{(A) }1
\qquad
\text{(B) }2
\qquad
\text{(C) }\sqrt 5
\qquad
\text{(D) }\sqrt 6
\qquad
\text{(E) }3
$

## Answer
√5
-/

-- # Formal Answer
noncomputable abbrev Q339.gt := fun (answer : ℝ) => answer = (√(5 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q339
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h₀ : 0 < a ∧ 0 < b)
  (h₁ : a ≠ b)
  (h₂ : abs (a - 1/a) = 1)
  (h₃ : abs (b - 1/b) = 1)
: (answer = a + b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q339
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h₀ : 0 < a ∧ 0 < b)
  (h₁ : a ≠ b)
  (h₂ : abs (a - 1/a) = 1)
  (h₃ : abs (b - 1/b) = 1)
: (answer = a + b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
