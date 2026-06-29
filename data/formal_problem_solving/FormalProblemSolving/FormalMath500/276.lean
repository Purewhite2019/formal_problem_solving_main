import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute $997^{-1}$ modulo $1000$. Express your answer as an integer from $0$ to $999$.

## Answer
333
-/

-- # Formal Answer
noncomputable abbrev Q276.gt := fun (answer : ℕ) => answer = (333 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q276
find (answer : ℕ) s.t.
: (997*answer ≡ 1 [MOD 1000] ∧ answer ≤ 999 ∧ answer ≥ 0)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q276
find_all (answer : ℕ) iff
: (997*answer ≡ 1 [MOD 1000] ∧ answer ≤ 999 ∧ answer ≥ 0)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
