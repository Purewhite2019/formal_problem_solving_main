import FormalProblemSolving.Basic
open Metric Set EuclideanGeometry

/-
## Problem
Find the length of the longest possible sequence of equal nonzero digits (in base 10) in which a perfect square can terminate. Also, find the smallest square that attains this length.

## Answer
(3, 1444)
-/

-- # Formal Answer
noncomputable abbrev Q18.gt := fun (answer : ℕ × ℕ) => answer = (((3 : ℕ), (1444 : ℕ)) : ℕ × ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q18
find (answer : ℕ × ℕ) s.t.
  (L : ℕ → ℕ)
  (hL : ∀ n : ℕ, L n ≤ (Nat.digits 10 n).length ∧
(∀ k : ℕ, k < L n → (Nat.digits 10 n)[k]! = (Nat.digits 10 n)[0]!) ∧
(L n ≠ (Nat.digits 10 n).length → (Nat.digits 10 n)[L n]! ≠ (Nat.digits 10 n)[0]!))
: ((∃ n : ℕ, (Nat.digits 10 (n^2))[0]! ≠ 0 ∧ L (n^2) = answer.1) ∧
(∀ n : ℕ, (Nat.digits 10 (n^2))[0]! ≠ 0 → L (n^2) ≤ answer.1) ∧
(∃ m : ℕ, m^2 = answer.2) ∧
L (answer.2) = answer.1 ∧
(Nat.digits 10 answer.2)[0]! ≠ 0 ∧
∀ n : ℕ, (Nat.digits 10 (n^2))[0]! ≠ 0 ∧ L (n^2) = answer.1 → n^2 ≥ answer.2)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q18
find_all (answer : ℕ × ℕ) iff
  (L : ℕ → ℕ)
  (hL : ∀ n : ℕ, L n ≤ (Nat.digits 10 n).length ∧
(∀ k : ℕ, k < L n → (Nat.digits 10 n)[k]! = (Nat.digits 10 n)[0]!) ∧
(L n ≠ (Nat.digits 10 n).length → (Nat.digits 10 n)[L n]! ≠ (Nat.digits 10 n)[0]!))
: ((∃ n : ℕ, (Nat.digits 10 (n^2))[0]! ≠ 0 ∧ L (n^2) = answer.1) ∧
(∀ n : ℕ, (Nat.digits 10 (n^2))[0]! ≠ 0 → L (n^2) ≤ answer.1) ∧
(∃ m : ℕ, m^2 = answer.2) ∧
L (answer.2) = answer.1 ∧
(Nat.digits 10 answer.2)[0]! ≠ 0 ∧
∀ n : ℕ, (Nat.digits 10 (n^2))[0]! ≠ 0 ∧ L (n^2) = answer.1 → n^2 ≥ answer.2)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
