import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $n$ is a multiple of three, what is the remainder when $(n + 4) + (n + 6) + (n + 8)$ is divided by $9$?

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q374.gt := fun (answer : ℕ) => answer = (0 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q374
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : 3 ∣ n)
: (answer = ((n + 4) + (n + 6) + (n + 8)) % 9)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q374
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : 3 ∣ n)
: (answer = ((n + 4) + (n + 6) + (n + 8)) % 9)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
