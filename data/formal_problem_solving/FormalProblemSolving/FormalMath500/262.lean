import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
There are finitely many primes $p$ for which the congruence $$8x\equiv 1\pmod{p}$$has no solutions $x$. Determine the sum of all such $p$.

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q262.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q262
find (answer : ℕ) s.t.
  (S : Finset ℕ)
  (hS : S = {p:ℕ | Nat.Prime p ∧ ¬∃ x : ℕ, 8 * x ≡ 1 [MOD p]})
: (answer = ∑ x ∈ S, x)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q262
find_all (answer : ℕ) iff
  (S : Finset ℕ)
  (hS : S = {p:ℕ | Nat.Prime p ∧ ¬∃ x : ℕ, 8 * x ≡ 1 [MOD p]})
: (answer = ∑ x ∈ S, x)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
