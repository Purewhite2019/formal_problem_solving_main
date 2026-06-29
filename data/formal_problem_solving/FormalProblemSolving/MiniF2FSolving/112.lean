import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute the least common multiple of $9999$ and $100{,}001$.

## Answer
90{,}900{,}909
-/

-- # Formal Answer
noncomputable abbrev Q112.gt := fun (answer : ℕ) => answer = (90900909 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q112
find (answer : ℕ) s.t.
: (answer = Nat.lcm 9999 100001)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q112
find_all (answer : ℕ) iff
: (answer = Nat.lcm 9999 100001)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
