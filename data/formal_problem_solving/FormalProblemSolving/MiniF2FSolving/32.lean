import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the modulo $7$ remainder of the sum $1+3+5+7+9+\dots+195+197+199.$

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q32.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q32
find (answer : ℕ) s.t.
  (v : ℕ → ℕ)
  (h₀ : ∀ n, v n = 2 * n - 1)
: (answer = (∑ k in Finset.Icc 1 100, v k) % 7)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q32
find_all (answer : ℕ) iff
  (v : ℕ → ℕ)
  (h₀ : ∀ n, v n = 2 * n - 1)
: (answer = (∑ k in Finset.Icc 1 100, v k) % 7)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
