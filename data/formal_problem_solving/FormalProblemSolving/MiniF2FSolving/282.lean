import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve the congruence $2n \equiv 15 \pmod{47}$, as a residue modulo 47.  (Give an answer between 0 and 46.)

## Answer
31
-/

-- # Formal Answer
noncomputable abbrev Q282.gt := fun (answer : ℕ) => answer = (31 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q282
find (answer : ℕ) s.t.
: (answer < 47) ∧ ((2 * answer) % 47 = 15)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q282
find_all (answer : ℕ) iff
: (answer < 47) ∧ ((2 * answer) % 47 = 15)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
