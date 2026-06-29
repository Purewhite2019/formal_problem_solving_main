import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $t=2s-s^2$ and $s=n^2 - 2^n+1$. What is the value of $t$ when $n=3$?

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q184.gt := fun (answer : ℝ) => answer = (0 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q184
find (answer : ℝ) s.t.
  (t : ℝ)
  (s : ℝ)
  (n : ℕ)
  (h₀ : t = 2 * s - s ^ 2)
  (h₁ : s = (n : ℝ) ^ 2 - (2 : ℝ) ^ n + 1)
  (hn : n = 3)
: (answer = t)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q184
find_all (answer : ℝ) iff
  (t : ℝ)
  (s : ℝ)
  (n : ℕ)
  (h₀ : t = 2 * s - s ^ 2)
  (h₁ : s = (n : ℝ) ^ 2 - (2 : ℝ) ^ n + 1)
  (hn : n = 3)
: (answer = t)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
