import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $x^3$ is a positive factor of $10!,$ how many possible integer values of $x$ are there?  (Reminder: For a positive integer $n$, the expression $n!$ stands for the product of the integers from 1 up to (and including) $n$.)

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q291.gt := fun (answer : ℕ) => answer = (6 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q291
find (answer : ℕ) s.t.
  (x : ℤ)
  (h₀ : x^3 > 0)
  (h₁ : x^3 ∣ Nat.factorial 10)
: (Nat.card {x : ℤ | x^3 > 0 ∧ x^3 ∣ Nat.factorial 10} = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q291
find_all (answer : ℕ) iff
  (x : ℤ)
  (h₀ : x^3 > 0)
  (h₁ : x^3 ∣ Nat.factorial 10)
: (Nat.card {x : ℤ | x^3 > 0 ∧ x^3 ∣ Nat.factorial 10} = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
