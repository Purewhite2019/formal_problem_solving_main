import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the greatest common divisor of $3339$, $2961$, and $1491$.

## Answer
21
-/

-- # Formal Answer
noncomputable abbrev Q250.gt := fun (answer : ℕ) => answer = (21 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q250
find (answer : ℕ) s.t.
: (answer = Nat.gcd 3339 (Nat.gcd 2961 1491))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q250
find_all (answer : ℕ) iff
: (answer = Nat.gcd 3339 (Nat.gcd 2961 1491))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
