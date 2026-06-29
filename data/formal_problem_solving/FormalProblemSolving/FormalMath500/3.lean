import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A bookcase has 3 shelves with a total of 24 books. The top shelf has 8 mystery books. The middle shelf has 10 math books. The bottom shelf has 6 science books. Two books are now taken off each shelf. What fraction of the books remaining on the three shelves are math books? Express your answer as a common fraction.

## Answer
\frac{4}{9}
-/

-- # Formal Answer
noncomputable abbrev Q3.gt := fun (answer : ℝ) => answer = (4 / 9 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q3
find (answer : ℝ) s.t.
: (answer = (10 - 2) / (24 - 3*2))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q3
find_all (answer : ℝ) iff
: (answer = (10 - 2) / (24 - 3*2))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
