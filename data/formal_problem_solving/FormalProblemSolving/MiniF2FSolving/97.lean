import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $F(a, b, c, d) = a^b + c ^ d$, what is the value of $b$ such that $F(4, b, 2, 3) = 12$?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q97.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q97
find (answer : ℝ) s.t.
  (F : ℝ → ℝ → ℝ → ℝ → ℝ)
  (h₀ : ∀ a b c d, F a b c d = a^b + c^d)
: (F 4 answer 2 3 = 12)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q97
find_all (answer : ℝ) iff
  (F : ℝ → ℝ → ℝ → ℝ → ℝ)
  (h₀ : ∀ a b c d, F a b c d = a^b + c^d)
: (F 4 answer 2 3 = 12)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
