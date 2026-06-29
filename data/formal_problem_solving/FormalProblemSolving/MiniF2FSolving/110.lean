import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let \[f(x) =
\begin{cases}
x^2+9 &\text{if }x<-5, \\
3x-8&\text{if }x\ge-5.
\end{cases}
\]If $f(x)=10$, find the sum of all possible values of $x$.

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q110.gt := fun (answer : ℝ) => answer = (6 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q110
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ x, x < -5 → f x = x ^ 2 + 9)
  (h₁ : ∀ x, x ≥ -5 → f x = 3 * x - 8)
  (S : Finset ℝ)
  (hS : ∀ x, x ∈ S ↔ f x = 10)
: (answer = ∑ x in S, x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q110
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ x, x < -5 → f x = x ^ 2 + 9)
  (h₁ : ∀ x, x ≥ -5 → f x = 3 * x - 8)
  (S : Finset ℝ)
  (hS : ∀ x, x ∈ S ↔ f x = 10)
: (answer = ∑ x in S, x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
