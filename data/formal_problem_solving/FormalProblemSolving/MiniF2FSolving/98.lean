import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the sixth term in the geometric sequence $\frac{27}{125}, \frac{9}{25}, \frac{3}{5},\ldots$?  Express your answer as a common fraction.

## Answer
25/9
-/

-- # Formal Answer
noncomputable abbrev Q98.gt := fun (answer : ℚ) => answer = (25 / 9 : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q98
find (answer : ℚ) s.t.
  (a : ℕ → ℚ)
  (h₀ : a 0 = 27 / 125)
  (h₁ : ∀ n, a (n + 1) = a n * (5 / 3))
: (answer = a 5)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q98
find_all (answer : ℚ) iff
  (a : ℕ → ℚ)
  (h₀ : a 0 = 27 / 125)
  (h₁ : ∀ n, a (n + 1) = a n * (5 / 3))
: (answer = a 5)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
