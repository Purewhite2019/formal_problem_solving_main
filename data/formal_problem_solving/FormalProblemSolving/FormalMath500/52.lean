import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If 4 daps = 7 yaps, and 5 yaps = 3 baps, how many daps equal 42 baps?

## Answer
40
-/

-- # Formal Answer
noncomputable abbrev Q52.gt := fun (answer : ℕ) => answer = (40 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q52
find (answer : ℕ) s.t.
  (daps : ℕ)
  (yaps : ℕ)
  (baps : ℕ)
  (h_daps_to_yaps : 4 * daps = 7 * yaps)
  (h_yaps_to_baps : 5 * yaps = 3 * baps)
: (answer * daps = 42 * baps)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q52
find_all (answer : ℕ) iff
  (daps : ℕ)
  (yaps : ℕ)
  (baps : ℕ)
  (h_daps_to_yaps : 4 * daps = 7 * yaps)
  (h_yaps_to_baps : 5 * yaps = 3 * baps)
: (answer * daps = 42 * baps)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
