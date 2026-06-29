import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $a$ and $b$ are positive integers such that $\gcd(a,b)=210$, $\mathop{\text{lcm}}[a,b]=210^3$, and $a<b$, how many possible values are there for $a$?

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q279.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q279
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {a : ℕ | 0 < a ∧ ∃ (b : ℕ), 0 < b ∧ a < b ∧ Nat.gcd a b = 210 ∧ Nat.lcm a b = 210^3})
: (answer = S.ncard)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q279
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {a : ℕ | 0 < a ∧ ∃ (b : ℕ), 0 < b ∧ a < b ∧ Nat.gcd a b = 210 ∧ Nat.lcm a b = 210^3})
: (answer = S.ncard)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
