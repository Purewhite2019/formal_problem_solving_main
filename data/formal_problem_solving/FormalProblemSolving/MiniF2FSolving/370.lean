import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Five plus $500\%$ of $10$ is the same as $110\%$ of what number?

## Answer
50
-/

-- # Formal Answer
noncomputable abbrev Q370.gt := fun (answer : ℝ) => answer = (50 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q370
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : 5 + 500 / 100 * 10 = 110 / 100 * x)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q370
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : 5 + 500 / 100 * 10 = 110 / 100 * x)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
