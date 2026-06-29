import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many values of $\theta$ in the interval $0<\theta\le 2\pi$ satisfy$1-3\sin\theta+5\cos3\theta = 0?$$\textbf{(A) }2 \qquad \textbf{(B) }4 \qquad \textbf{(C) }5\qquad \textbf{(D) }6 \qquad \textbf{(E) }8$

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q347.gt := fun (answer : ℕ) => answer = (6 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q347
find (answer : ℕ) s.t.
  (S : Finset ℝ)
  (h₀ : ∀ (x : ℝ), x ∈ S ↔ 0 < x ∧ x ≤ 2 * Real.pi ∧ 1 - 3 * Real.sin x + 5 * Real.cos (3 * x) = 0)
: (answer = S.card)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q347
find_all (answer : ℕ) iff
  (S : Finset ℝ)
  (h₀ : ∀ (x : ℝ), x ∈ S ↔ 0 < x ∧ x ≤ 2 * Real.pi ∧ 1 - 3 * Real.sin x + 5 * Real.cos (3 * x) = 0)
: (answer = S.card)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
