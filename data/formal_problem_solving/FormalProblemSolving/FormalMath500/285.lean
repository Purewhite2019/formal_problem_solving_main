import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $n \equiv 2 \pmod{7}$, then find the remainder when $(n + 2)(n + 4)(n + 6)$ is divided by 7.

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q285.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q285
find (answer : ℕ) s.t.
  (n : ℕ)
  (h : n % 7 = 2)
: (answer = ((n + 2) * (n + 4) * (n + 6)) % 7)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q285
find_all (answer : ℕ) iff
  (n : ℕ)
  (h : n % 7 = 2)
: (answer = ((n + 2) * (n + 4) * (n + 6)) % 7)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
