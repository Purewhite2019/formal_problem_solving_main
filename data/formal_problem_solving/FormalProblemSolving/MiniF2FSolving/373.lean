import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Assuming $x\ne0$, simplify $\frac{12}{x \cdot x} \cdot \frac{x^4}{14x}\cdot \frac{35}{3x}$.

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q373.gt := fun (answer : ℝ) => answer = (10 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q373
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : x ≠ 0)
: (answer = 12 / (x * x) * (x^4 / (14 * x)) * (35 / (3 * x)))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q373
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : x ≠ 0)
: (answer = 12 / (x * x) * (x^4 / (14 * x)) * (35 / (3 * x)))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
