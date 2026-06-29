import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $10^x - 10 = 9990,$ what is $x$ equal to?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q47.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q47
find (answer : ℕ) s.t.
  (x : ℕ)
  (h₀ : 10^x - 10 = 9990)
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q47
find_all (answer : ℕ) iff
  (x : ℕ)
  (h₀ : 10^x - 10 = 9990)
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
