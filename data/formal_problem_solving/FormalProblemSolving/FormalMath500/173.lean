import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sum of the first $N$ positive odd numbers is 121. What is the value of $N$?

## Answer
11
-/

-- # Formal Answer
noncomputable abbrev Q173.gt := fun (answer : ℕ) => answer = (11 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q173
find (answer : ℕ) s.t.
  (N : ℕ)
  (h₀ : N > 0)
  (h₁ : ∑ k in Finset.range N, (2 * k + 1) = 121)
: (N = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q173
find_all (answer : ℕ) iff
  (N : ℕ)
  (h₀ : N > 0)
  (h₁ : ∑ k in Finset.range N, (2 * k + 1) = 121)
: (N = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
