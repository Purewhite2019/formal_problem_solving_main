import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute $99^2+99+1$ in your head.

## Answer
9901
-/

-- # Formal Answer
noncomputable abbrev Q221.gt := fun (answer : ℕ) => answer = (9901 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q221
find (answer : ℕ) s.t.
  (result : ℕ)
  (h₀ : result = 99^2 + 99 + 1)
: (result = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q221
find_all (answer : ℕ) iff
  (result : ℕ)
  (h₀ : result = 99^2 + 99 + 1)
: (result = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
