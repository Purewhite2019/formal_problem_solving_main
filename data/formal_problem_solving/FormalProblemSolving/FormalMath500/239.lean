import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What's the largest eight-digit base 2 integer?  Express your answer in base 10.

## Answer
255
-/

-- # Formal Answer
noncomputable abbrev Q239.gt := fun (answer : ℕ) => answer = (255 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q239
find (answer : ℕ) s.t.
: (answer = Nat.ofDigits 2 (List.replicate 8 1))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q239
find_all (answer : ℕ) iff
: (answer = Nat.ofDigits 2 (List.replicate 8 1))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
