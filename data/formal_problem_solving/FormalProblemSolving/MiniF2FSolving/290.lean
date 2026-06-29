import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose that the roots of the polynomial $P(x)=x^3+ax^2+bx+c$ are $\cos \frac{2\pi}7,\cos \frac{4\pi}7,$ and $\cos \frac{6\pi}7$, where angles are in radians. What is $abc$?

$\textbf{(A) }{-}\frac{3}{49} \qquad \textbf{(B) }{-}\frac{1}{28} \qquad \textbf{(C) }\frac{\sqrt[3]7}{64} \qquad \textbf{(D) }\frac{1}{32}\qquad \textbf{(E) }\frac{1}{28}$

## Answer
$\frac{1}{32}$
-/

-- # Formal Answer
noncomputable abbrev Q290.gt := fun (answer : ℝ) => answer = (1 / 32 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q290
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = x^3 + a * x^2 + b * x + c)
  (h₁ : f⁻¹' {0} = {Real.cos (2 * Real.pi / 7), Real.cos (4 * Real.pi / 7), Real.cos (6 * Real.pi / 7)})
: (answer = a * b * c)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q290
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = x^3 + a * x^2 + b * x + c)
  (h₁ : f⁻¹' {0} = {Real.cos (2 * Real.pi / 7), Real.cos (4 * Real.pi / 7), Real.cos (6 * Real.pi / 7)})
: (answer = a * b * c)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
