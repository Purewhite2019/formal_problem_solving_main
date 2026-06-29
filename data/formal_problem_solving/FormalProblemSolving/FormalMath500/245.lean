import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $n$ be a positive integer.  What is the greatest possible value of $\gcd(n + 7, 2n + 1)$?

## Answer
13
-/

-- # Formal Answer
noncomputable abbrev Q245.gt := fun (answer : ℕ) => answer = (13 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q245
find (answer : ℕ) s.t.
  (n : ℕ)
: (∀ (n : ℕ), Nat.gcd (n + 7) (2 * n + 1) ≤ answer) ∧ (∃ (n : ℕ), Nat.gcd (n + 7) (2 * n + 1) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q245
find_all (answer : ℕ) iff
  (n : ℕ)
: (∀ (n : ℕ), Nat.gcd (n + 7) (2 * n + 1) ≤ answer) ∧ (∃ (n : ℕ), Nat.gcd (n + 7) (2 * n + 1) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
