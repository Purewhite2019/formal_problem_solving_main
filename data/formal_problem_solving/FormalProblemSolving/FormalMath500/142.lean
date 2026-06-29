import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Consider the geometric sequence $\frac{125}{9}, \frac{25}{3}, 5, 3, \ldots$. What is the eighth term of the sequence? Express your answer as a common fraction.

## Answer
\frac{243}{625}
-/

-- # Formal Answer
noncomputable abbrev Q142.gt := fun (answer : ℚ) => answer = (243 / 625 : ℚ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q142
find (answer : ℚ) s.t.
  (a : ℕ → ℚ)
  (r : ℚ)
  (h₀ : a 1 = (125:ℚ) / 9)
  (h₁ : a 2 = (25:ℚ) / 3)
  (h₂ : a 3 = 5)
  (h₃ : a 4 = 3)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * r)
: (a 8 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q142
find_all (answer : ℚ) iff
  (a : ℕ → ℚ)
  (r : ℚ)
  (h₀ : a 1 = (125:ℚ) / 9)
  (h₁ : a 2 = (25:ℚ) / 3)
  (h₂ : a 3 = 5)
  (h₃ : a 4 = 3)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * r)
: (a 8 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
