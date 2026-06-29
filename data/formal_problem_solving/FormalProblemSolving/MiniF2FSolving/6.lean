import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose $m$ is a two-digit positive integer such that $6^{-1}\pmod m$ exists and $6^{-1}\equiv 6^2\pmod m$. What is $m$?

## Answer
43
-/

-- # Formal Answer
noncomputable abbrev Q6.gt := fun (answer : ℕ) => answer = (43 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q6
find (answer : ℕ) s.t.
  (m : ℕ)
  (h_left : 10 ≤ m)
  (h_right : m ≤ 99)
  (h_coprime : Nat.gcd 6 m = 1)
  (h_inv_eq : (6 * (6 ^ 2)) % m = 1)
: (answer = m)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q6
find_all (answer : ℕ) iff
  (m : ℕ)
  (h_left : 10 ≤ m)
  (h_right : m ≤ 99)
  (h_coprime : Nat.gcd 6 m = 1)
  (h_inv_eq : (6 * (6 ^ 2)) % m = 1)
: (answer = m)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
