import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute $17^{-1}\pmod{83}$. Express your answer as a residue from $0$ to $82$, inclusive.

(You may find it helpful to consider the fact that $17\cdot 5=85$.)

## Answer
44
-/

-- # Formal Answer
noncomputable abbrev Q256.gt := fun (answer : ℕ) => answer = (44 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q256
find (answer : ℕ) s.t.
: (17 * answer ≡ 1 [MOD 83] ∧ answer ≤ 82 ∧ answer ≥ 0)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q256
find_all (answer : ℕ) iff
: (17 * answer ≡ 1 [MOD 83] ∧ answer ≤ 82 ∧ answer ≥ 0)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
