import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The graph of $f(x)=\frac{2x}{x^2-5x-14}$ has vertical asymptotes $x=a$ and $x=b$, and horizontal asymptote $y=c$.  Find $a+b+c$.

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q180.gt := fun (answer : ℝ) => answer = (5 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q180
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : ∀ x, x ≠ a → x ≠ b → (x^2 - 5 * x - 14) ≠ 0)
  (h₁ : a ≠ b)
  (h₂ : a^2 - 5 * a - 14 = 0)
  (h₃ : b^2 - 5 * b - 14 = 0)
  (h₄ : c = 0)
: (a + b + c = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q180
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : ∀ x, x ≠ a → x ≠ b → (x^2 - 5 * x - 14) ≠ 0)
  (h₁ : a ≠ b)
  (h₂ : a^2 - 5 * a - 14 = 0)
  (h₃ : b^2 - 5 * b - 14 = 0)
  (h₄ : c = 0)
: (a + b + c = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
