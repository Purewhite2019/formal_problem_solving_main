import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the sum of the two values of $x$ for which $(x+3)^2 = 121$?

## Answer
-6
-/

-- # Formal Answer
noncomputable abbrev Q239.gt := fun (answer : ℝ) => answer = (-6 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q239
find (answer : ℝ) s.t.
  (S : Finset ℝ)
  (h₀ : ∀ (x : ℝ), x ∈ S ↔ (x + 3)^2 = 121)
: (answer = ∑ k in S, k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q239
find_all (answer : ℝ) iff
  (S : Finset ℝ)
  (h₀ : ∀ (x : ℝ), x ∈ S ↔ (x + 3)^2 = 121)
: (answer = ∑ k in S, k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
