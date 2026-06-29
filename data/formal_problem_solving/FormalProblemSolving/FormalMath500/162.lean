import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If the fourth term of an arithmetic sequence is $200$ and the eighth term is $500$, what is the sixth term?

## Answer
350
-/

-- # Formal Answer
noncomputable abbrev Q162.gt := fun (answer : ℝ) => answer = (350 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q162
find (answer : ℝ) s.t.
  (a₁ : ℝ)
  (d : ℝ)
  (h₀ : a₁ + 3 * d = 200)
  (h₁ : a₁ + 7 * d = 500)
: (a₁ + 5 * d = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q162
find_all (answer : ℝ) iff
  (a₁ : ℝ)
  (d : ℝ)
  (h₀ : a₁ + 3 * d = 200)
  (h₁ : a₁ + 7 * d = 500)
: (a₁ + 5 * d = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
