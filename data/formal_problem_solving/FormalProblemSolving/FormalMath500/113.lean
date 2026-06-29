import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sum of 27 consecutive positive integers is $3^7$. What is their median?

## Answer
81
-/

-- # Formal Answer
noncomputable abbrev Q113.gt := fun (answer : ℕ) => answer = (81 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q113
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : n > 0)
  (h₁ : ∑ k in Finset.range 27, (n + k) = 3^7)
: (n + 13 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q113
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : n > 0)
  (h₁ : ∑ k in Finset.range 27, (n + k) = 3^7)
: (n + 13 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
