import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The symbols $\triangle$, $\square$, $\diamond$, $\clubsuit$ represent four different integers from 1 to 9. Using the equations below, what is the value of   $\square$ ? \begin{align*}
\triangle + \square &= \clubsuit \\
\triangle + \triangle &= \diamond +\diamond + \diamond + \diamond + \diamond \\
\triangle + \triangle &= \clubsuit + \diamond.
\end{align*}

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q138.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q138
find (answer : ℕ) s.t.
  (triangle : ℕ)
  (square : ℕ)
  (diamond : ℕ)
  (club : ℕ)
  (h₀ : triangle ≠ square ∧ triangle ≠ diamond ∧ triangle ≠ club ∧
        square ≠ diamond ∧ square ≠ club ∧ diamond ≠ club)
  (h₁ : triangle ≥ 1 ∧ triangle ≤ 9 ∧ square ≥ 1 ∧ square ≤ 9 ∧
        diamond ≥ 1 ∧ diamond ≤ 9 ∧ club ≥ 1 ∧ club ≤ 9)
  (h₂ : triangle + square = club)
  (h₃ : triangle + triangle = diamond + diamond + diamond + diamond + diamond)
  (h₄ : triangle + triangle = club + diamond)
: (square = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q138
find_all (answer : ℕ) iff
  (triangle : ℕ)
  (square : ℕ)
  (diamond : ℕ)
  (club : ℕ)
  (h₀ : triangle ≠ square ∧ triangle ≠ diamond ∧ triangle ≠ club ∧
        square ≠ diamond ∧ square ≠ club ∧ diamond ≠ club)
  (h₁ : triangle ≥ 1 ∧ triangle ≤ 9 ∧ square ≥ 1 ∧ square ≤ 9 ∧
        diamond ≥ 1 ∧ diamond ≤ 9 ∧ club ≥ 1 ∧ club ≤ 9)
  (h₂ : triangle + square = club)
  (h₃ : triangle + triangle = diamond + diamond + diamond + diamond + diamond)
  (h₄ : triangle + triangle = club + diamond)
: (square = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
