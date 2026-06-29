import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the sum of all solutions of the equation $|2-x|= 3$.

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q331.gt := fun (answer : ℝ) => answer = (4 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q331
find (answer : ℝ) s.t.
  (S : Finset ℝ)
  (h₀ : ∀ (x : ℝ), x ∈ S ↔ |2 - x| = 3)
: (answer = ∑ k in S, k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q331
find_all (answer : ℝ) iff
  (S : Finset ℝ)
  (h₀ : ∀ (x : ℝ), x ∈ S ↔ |2 - x| = 3)
: (answer = ∑ k in S, k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
