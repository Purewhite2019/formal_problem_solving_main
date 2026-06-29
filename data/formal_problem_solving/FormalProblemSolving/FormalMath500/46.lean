import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For what value of $x$ is $2^3\cdot3^x=72$?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q46.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q46
find (answer : ℕ) s.t.
  (x : ℕ)
  (h₀ : 2^3 * 3^x = 72)
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q46
find_all (answer : ℕ) iff
  (x : ℕ)
  (h₀ : 2^3 * 3^x = 72)
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
