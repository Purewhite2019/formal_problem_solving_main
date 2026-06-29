import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find $24^{-1} \pmod{11^2}$. That is, find the residue $b$ for which $24b \equiv 1\pmod{11^2}$.

Express your answer as an integer from $0$ to $11^2-1$, inclusive.

## Answer
116
-/

-- # Formal Answer
noncomputable abbrev Q283.gt := fun (answer : ZMod (11^2)) => answer = (116 : ZMod ((11 : ℕ) ^ (2 : ℕ) : ℕ))

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q283
find (answer : ZMod (11^2)) s.t.
: (answer = 24⁻¹)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q283
find_all (answer : ZMod (11^2)) iff
: (answer = 24⁻¹)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
