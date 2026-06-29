import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $$1+12+123+1234+12345+123456+1234567+12345678$$is congruent to $n$ modulo 9, where $0\le n<9$, what is the value of $n$?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q246.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q246
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : 0 ≤ n ∧ n < 9)
  (h₁ : (1 + 12 + 123 + 1234 + 12345 + 123456 + 1234567 + 12345678) ≡ n [MOD 9])
: (answer  = n)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q246
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : 0 ≤ n ∧ n < 9)
  (h₁ : (1 + 12 + 123 + 1234 + 12345 + 123456 + 1234567 + 12345678) ≡ n [MOD 9])
: (answer  = n)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
