import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is $\left(4\dfrac{5}{8}\right)^{55} \cdot \left(\dfrac{8}{37}\right)^{55}$?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q63.gt := fun (answer : ℚ) => answer = (1 : ℚ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q63
find (answer : ℚ) s.t.
  (a : ℚ)
  (b : ℚ)
  (hab : a = 4 + 5/8 ∧ b = 8/37)
: ((a^55) * (b^55) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q63
find_all (answer : ℚ) iff
  (a : ℚ)
  (b : ℚ)
  (hab : a = 4 + 5/8 ∧ b = 8/37)
: ((a^55) * (b^55) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
