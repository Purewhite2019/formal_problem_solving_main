import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find an integer $n$ such that $0\leq n<398$ and $n$ is a multiplicative inverse to 7 modulo 398.

## Answer
57
-/

-- # Formal Answer
noncomputable abbrev Q19.gt := fun (answer : ℕ) => answer = (57 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q19
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : n < 398)
  (h₁ : (n * 7) % 398 = 1)
: (answer = n)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q19
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : n < 398)
  (h₁ : (n * 7) % 398 = 1)
: (answer = n)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
