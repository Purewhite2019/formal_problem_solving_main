import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
I have a bag with yellow and blue marbles in it. At the moment, the ratio of blue marbles to yellow marbles is 4:3. If I add 5 blue marbles and remove 3 yellow marbles, the ratio will be 7:3. How many blue marbles were in the bag before I added more?

## Answer
16
-/

-- # Formal Answer
noncomputable abbrev Q198.gt := fun (answer : ℕ) => answer = (16 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q198
find (answer : ℕ) s.t.
  (blue : ℕ)
  (yellow : ℕ)
  (h₀ : 3 * blue = 4 * yellow)
  (h₁ : 3 * (blue + 5) = 7 * (yellow - 3))
: (blue = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q198
find_all (answer : ℕ) iff
  (blue : ℕ)
  (yellow : ℕ)
  (h₀ : 3 * blue = 4 * yellow)
  (h₁ : 3 * (blue + 5) = 7 * (yellow - 3))
: (blue = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
