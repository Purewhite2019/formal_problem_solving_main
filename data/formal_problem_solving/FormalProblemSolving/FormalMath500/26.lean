import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Remmy wants to divide $10$ by $\frac{2}{3}$, but he cannot remember how to do that.  By what number should he multiply $10$ to get the answer?

## Answer
\frac{3}{2}
-/

-- # Formal Answer
noncomputable abbrev Q26.gt := fun (answer : ℚ) => answer = (3 / 2 : ℚ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q26
find (answer : ℚ) s.t.
  (n : ℚ)
  (divisor : ℚ)
  (hn : n = 10 ∧ divisor = 2 / 3)
: (n * answer = n / divisor)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q26
find_all (answer : ℚ) iff
  (n : ℚ)
  (divisor : ℚ)
  (hn : n = 10 ∧ divisor = 2 / 3)
: (n * answer = n / divisor)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
