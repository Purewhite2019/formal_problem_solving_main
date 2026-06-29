import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What fraction is the same as \[
\frac{2-4+6-8+10-12+14}{3-6+9-12+15-18+21}?
\]

## Answer
2/3
-/

-- # Formal Answer
noncomputable abbrev Q85.gt := fun (answer : ℚ) => answer = (2 / 3 : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q85
find (answer : ℚ) s.t.
  (q : ℚ)
  (p : ℚ)
  (h₀ : q = 2 - 4 + 6 - 8 + 10 - 12 + 14)
  (h₁ : p = 3 - 6 + 9 - 12 + 15 - 18 + 21)
: (answer = q / p)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q85
find_all (answer : ℚ) iff
  (q : ℚ)
  (p : ℚ)
  (h₀ : q = 2 - 4 + 6 - 8 + 10 - 12 + 14)
  (h₁ : p = 3 - 6 + 9 - 12 + 15 - 18 + 21)
: (answer = q / p)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
