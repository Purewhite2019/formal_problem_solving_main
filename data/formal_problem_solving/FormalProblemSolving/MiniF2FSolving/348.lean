import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
When the two-digit integer $``\text{AB}"$ is cubed, the value is $912,\!673$. What is $A + B$?

## Answer
16
-/

-- # Formal Answer
noncomputable abbrev Q348.gt := fun (answer : ℕ) => answer = (16 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q348
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (h₀ : 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9)
  (h₁ : (10 * a + b)^3 = 912673)
: (answer = a + b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q348
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (h₀ : 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9)
  (h₁ : (10 * a + b)^3 = 912673)
: (answer = a + b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
