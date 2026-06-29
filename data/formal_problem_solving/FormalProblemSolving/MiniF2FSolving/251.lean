import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve the congruence $3n \equiv 2 \pmod{11}$, as a residue modulo 11.  (Give an answer between 0 and 10.)

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q251.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q251
find (answer : ℕ) s.t.
: (answer < 11) ∧ ((3 * answer) % 11 = 2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q251
find_all (answer : ℕ) iff
: (answer < 11) ∧ ((3 * answer) % 11 = 2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
