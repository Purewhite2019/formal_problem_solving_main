import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $x,y,$ and $z$ are positive numbers satisfying

$x + \frac{1}{y} = 4,\qquad y + \frac{1}{z} = 1, \qquad \text{and} \qquad z + \frac{1}{x} = \frac{7}{3}$

Then what is the value of $xyz$ ?

$\text {(A)}\ \frac{2}{3} \qquad \text {(B)}\ 1 \qquad \text {(C)}\ \frac{4}{3} \qquad \text {(D)}\ 2 \qquad \text {(E)}\ \frac{7}{3}$

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q360.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q360
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h₀ : 0 < x ∧ 0 < y ∧ 0 < z)
  (h₁ : x + 1/y = 4)
  (h₂ : y + 1/z = 1)
  (h₃ : z + 1/x = 7/3)
: (answer = x * y * z)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q360
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h₀ : 0 < x ∧ 0 < y ∧ 0 < z)
  (h₁ : x + 1/y = 4)
  (h₂ : y + 1/z = 1)
  (h₃ : z + 1/x = 7/3)
: (answer = x * y * z)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
