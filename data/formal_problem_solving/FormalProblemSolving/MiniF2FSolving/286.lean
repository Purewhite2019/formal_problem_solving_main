import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
We write $\lfloor X \rfloor$ to mean the greatest integer less than or equal to $X$; for example $\lfloor 3\frac{1}{2} \rfloor = 3$. If $N = \frac{1}{3}$, what is the value of $\lfloor 10N \rfloor + \lfloor 100N \rfloor + \lfloor 1000N \rfloor + \lfloor 10,000N \rfloor$?

## Answer
3702
-/

-- # Formal Answer
noncomputable abbrev Q286.gt := fun (answer : ℤ) => answer = (3702 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q286
find (answer : ℤ) s.t.
  (N : ℝ)
  (h₀ : N = 1 / 3)
: (answer = Int.floor (10 * N) + Int.floor (100 * N) + Int.floor (1000 * N) + Int.floor (10000 * N))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q286
find_all (answer : ℤ) iff
  (N : ℝ)
  (h₀ : N = 1 / 3)
: (answer = Int.floor (10 * N) + Int.floor (100 * N) + Int.floor (1000 * N) + Int.floor (10000 * N))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
