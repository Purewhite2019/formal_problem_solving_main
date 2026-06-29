import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the modulo $7$ remainder of the sum $1+3+5+7+9+\dots+195+197+199.$

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q266.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q266
find (answer : ℕ) s.t.
: ((∑ n ∈ (Finset.range 100), (2*n + 1) ) % 7= answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q266
find_all (answer : ℕ) iff
: ((∑ n ∈ (Finset.range 100), (2*n + 1) ) % 7= answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
