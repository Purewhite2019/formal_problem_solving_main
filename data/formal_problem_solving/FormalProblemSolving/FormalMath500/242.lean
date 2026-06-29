import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
On a particular map, $3$ inches on the map equates to $10$ miles in real life. If you know that the real life distance between two buildings on the map is $53.25$ miles, what would the distance between the buildings be (in inches) on the map, expressed as a fraction?

## Answer
\frac{639}{40}
-/

-- # Formal Answer
noncomputable abbrev Q242.gt := fun (answer : ℝ) => answer = ((↑(639 : ℚ) : ℝ) / (40 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q242
find (answer : ℝ) s.t.
: (answer = (53.25 * 3) / 10)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q242
find_all (answer : ℝ) iff
: (answer = (53.25 * 3) / 10)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
