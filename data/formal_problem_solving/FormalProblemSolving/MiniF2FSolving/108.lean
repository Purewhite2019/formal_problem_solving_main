import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f$ be a linear function for which $f(6) - f(2) = 12.$ What is $f(12) - f(2)?$

$
\text {(A) } 12 \qquad \text {(B) } 18 \qquad \text {(C) } 24 \qquad \text {(D) } 30 \qquad \text {(E) } 36
$

## Answer
30
-/

-- # Formal Answer
noncomputable abbrev Q108.gt := fun (answer : ℝ) => answer = (30 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q108
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = a * x + b)
  (h₁ : f 6 - f 2 = 12)
: (answer = f 12 - f 2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q108
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = a * x + b)
  (h₁ : f 6 - f 2 = 12)
: (answer = f 12 - f 2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
