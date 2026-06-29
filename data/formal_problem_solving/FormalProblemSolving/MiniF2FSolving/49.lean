import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let \[f(x) = \begin{cases}
-x^2 - 1 &\text{if }x<0, \\
2&\text{if }0 \le x< 4, \\
\sqrt{x}&\text{if }x \ge 4.
\end{cases}
\]Find $f(\pi)$.

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q49.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q49
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ x < 0, f x = -(x^2) - 1)
  (h₁ : ∀ x, 0 ≤ x ∧ x < 4 → f x = 2)
  (h₂ : ∀ x ≥ 4, f x = Real.sqrt x)
: (answer = f π)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q49
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ x < 0, f x = -(x^2) - 1)
  (h₁ : ∀ x, 0 ≤ x ∧ x < 4 → f x = 2)
  (h₂ : ∀ x ≥ 4, f x = Real.sqrt x)
: (answer = f π)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
