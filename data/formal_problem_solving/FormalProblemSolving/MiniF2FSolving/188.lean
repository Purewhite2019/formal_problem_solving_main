import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Together, Larry and Lenny have $\$$35. Larry has two-fifths of Lenny's amount. How many more dollars than Larry does Lenny have?

## Answer
15
-/

-- # Formal Answer
noncomputable abbrev Q188.gt := fun (answer : ℝ) => answer = (15 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q188
find (answer : ℝ) s.t.
  (Larry : ℝ)
  (Lenny : ℝ)
  (h₀ : 0 < Larry ∧ 0 < Lenny)
  (h₁ : Larry + Lenny = 35)
  (h₂ : Larry = (2/5) * Lenny)
: (answer = Lenny - Larry)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q188
find_all (answer : ℝ) iff
  (Larry : ℝ)
  (Lenny : ℝ)
  (h₀ : 0 < Larry ∧ 0 < Lenny)
  (h₁ : Larry + Lenny = 35)
  (h₂ : Larry = (2/5) * Lenny)
: (answer = Lenny - Larry)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
