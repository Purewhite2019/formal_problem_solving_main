import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given that $a,$ $b,$ $c,$ $d,$ $e$ are real numbers such that
\begin{align*}
a + b + c + d + e &= 8, \\
a^2 + b^2 + c^2 + d^2 + e^2 &= 16.
\end{align*}Determine the maximum value of $e.$

## Answer
\frac{16}{5}
-/

-- # Formal Answer
noncomputable abbrev Q356.gt := fun (answer : ℝ) => answer = ((↑(16 : ℚ) : ℝ) / (5 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q356
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (e : ℝ)
  (h₀ : a + b + c + d + e = 8)
  (h₁ : a^2 + b^2 + c^2 + d^2 + e^2 = 16)
: ((∀ (a b c d e : ℝ), (a + b + c + d + e = 8 ∧ a^2 + b^2 + c^2 + d^2 + e^2 = 16) → e ≤ answer) ∧
  (∃ (a b c d e : ℝ), (a + b + c + d + e = 8 ∧ a^2 + b^2 + c^2 + d^2 + e^2 = 16) ∧ e = answer))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q356
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (e : ℝ)
  (h₀ : a + b + c + d + e = 8)
  (h₁ : a^2 + b^2 + c^2 + d^2 + e^2 = 16)
: ((∀ (a b c d e : ℝ), (a + b + c + d + e = 8 ∧ a^2 + b^2 + c^2 + d^2 + e^2 = 16) → e ≤ answer) ∧
  (∃ (a b c d e : ℝ), (a + b + c + d + e = 8 ∧ a^2 + b^2 + c^2 + d^2 + e^2 = 16) ∧ e = answer))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
