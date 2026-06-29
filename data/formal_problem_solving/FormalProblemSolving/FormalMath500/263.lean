import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
One gear turns $33\frac{1}{3}$ times in a minute.  Another gear turns   45 times in a minute.  Initially, a mark on each gear is pointing  due north.  After how many seconds will the two gears next  have both their marks pointing due north?

## Answer
36
-/

-- # Formal Answer
noncomputable abbrev Q263.gt := fun (answer : ℕ) => answer = (36 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q263
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {x : ℕ| x > 0 ∧ (3*60) ∣ (x * 100)  ∧ 60∣ (x * 45)})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q263
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {x : ℕ| x > 0 ∧ (3*60) ∣ (x * 100)  ∧ 60∣ (x * 45)})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
