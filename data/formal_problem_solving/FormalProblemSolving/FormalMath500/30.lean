import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Rick is thinking of a positive factor of $14$ and Steve is thinking of a positive factor of $42$.  If Rick and Steve are thinking of the same number, how many possible numbers could they be thinking of?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q30.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q30
find (answer : ℕ) s.t.
: (answer = (Nat.divisors (Nat.gcd 14 42)).card)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q30
find_all (answer : ℕ) iff
: (answer = (Nat.divisors (Nat.gcd 14 42)).card)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
