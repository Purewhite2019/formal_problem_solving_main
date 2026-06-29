import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $n$ be the number of integers $m$ in the range $1\le m\le 8$ such that $\text{gcd}(m,8)=1$. What is the remainder when $3^n$ is divided by $8$?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q96.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q96
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : n = Finset.card (Finset.filter (λ x => Nat.gcd x 8 = 1) (Finset.Icc 1 8)))
: (answer = (3^n) % 8)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q96
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : n = Finset.card (Finset.filter (λ x => Nat.gcd x 8 = 1) (Finset.Icc 1 8)))
: (answer = (3^n) % 8)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
