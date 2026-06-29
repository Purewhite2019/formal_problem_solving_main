import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The two positive integer solutions of the equation $x^2 - mx + n = 0$ are $k$ and $t$, where $m$ and $n$ are both prime numbers and $k > t$. What is the value of $m^n + n^m + k^t + t^k$?

## Answer
20
-/

-- # Formal Answer
noncomputable abbrev Q340.gt := fun (answer : ℕ) => answer = (20 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q340
find (answer : ℕ) s.t.
  (k : ℕ)
  (t : ℕ)
  (m : ℕ)
  (n : ℕ)
  (h_pos_k : 0 < k)
  (h_pos_t : 0 < t)
  (h₀ : Nat.Prime m ∧ Nat.Prime n)
  (h₁ : t < k)
  (h₂ : (k : ℤ)^2 - (m : ℤ) * (k : ℤ) + (n : ℤ) = 0)
  (h₃ : (t : ℤ)^2 - (m : ℤ) * (t : ℤ) + (n : ℤ) = 0)
: (answer = m^n + n^m + k^t + t^k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q340
find_all (answer : ℕ) iff
  (k : ℕ)
  (t : ℕ)
  (m : ℕ)
  (n : ℕ)
  (h_pos_k : 0 < k)
  (h_pos_t : 0 < t)
  (h₀ : Nat.Prime m ∧ Nat.Prime n)
  (h₁ : t < k)
  (h₂ : (k : ℤ)^2 - (m : ℤ) * (k : ℤ) + (n : ℤ) = 0)
  (h₃ : (t : ℤ)^2 - (m : ℤ) * (t : ℤ) + (n : ℤ) = 0)
: (answer = m^n + n^m + k^t + t^k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
