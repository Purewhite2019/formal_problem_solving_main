import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Riproarin' Ringo was roping a recalcitrant dogie. Ringo decided to give the dogie a reprieve by calculating \[|(1-i)^8|\]before riding after the dogie. What answer should Ringo have found?

## Answer
16
-/

-- # Formal Answer
noncomputable abbrev Q330.gt := fun (answer : ℝ) => answer = (16 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q330
find (answer : ℝ) s.t.
: (answer = Complex.abs ((1-Complex.I)^8))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q330
find_all (answer : ℝ) iff
: (answer = Complex.abs ((1-Complex.I)^8))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
