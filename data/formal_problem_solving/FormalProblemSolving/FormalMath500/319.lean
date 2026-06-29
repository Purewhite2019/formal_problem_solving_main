import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given that \begin{align*}x_{1}&=211,\\ x_{2}&=375,\\ x_{3}&=420,\\ x_{4}&=523,\ \text{and}\\ x_{n}&=x_{n-1}-x_{n-2}+x_{n-3}-x_{n-4}\ \text{when}\ n\geq5, \end{align*}find the value of $x_{531}+x_{753}+x_{975}$.

## Answer
898
-/

-- # Formal Answer
noncomputable abbrev Q319.gt := fun (answer : ℤ) => answer = (898 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q319
find (answer : ℤ) s.t.
  (x : ℕ → ℤ)
  (h₀ : x 1 = 211)
  (h₁ : x 2 = 375)
  (h₂ : x 3 = 420)
  (h₃ : x 4 = 523)
  (h₄ : ∀ n ≥ 5, x n = x (n - 1) - x (n - 2) + x (n - 3) - x (n - 4))
: (answer = x 531 + x 753 + x 975)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q319
find_all (answer : ℤ) iff
  (x : ℕ → ℤ)
  (h₀ : x 1 = 211)
  (h₁ : x 2 = 375)
  (h₂ : x 3 = 420)
  (h₃ : x 4 = 523)
  (h₄ : ∀ n ≥ 5, x n = x (n - 1) - x (n - 2) + x (n - 3) - x (n - 4))
: (answer = x 531 + x 753 + x 975)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
