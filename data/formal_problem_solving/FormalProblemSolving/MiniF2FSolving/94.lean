import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $n$ is a positive integer such that $2n$ has 28 positive divisors and $3n$ has 30 positive divisors, then how many positive divisors does $6n$ have?

## Answer
35
-/

-- # Formal Answer
noncomputable abbrev Q94.gt := fun (answer : ℕ) => answer = (35 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q94
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : 0 < n)
  (h₁ : Finset.card (Nat.divisors (2 * n)) = 28)
  (h₂ : Finset.card (Nat.divisors (3 * n)) = 30)
: (answer = Finset.card (Nat.divisors (6 * n)))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q94
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : 0 < n)
  (h₁ : Finset.card (Nat.divisors (2 * n)) = 28)
  (h₂ : Finset.card (Nat.divisors (3 * n)) = 30)
: (answer = Finset.card (Nat.divisors (6 * n)))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
