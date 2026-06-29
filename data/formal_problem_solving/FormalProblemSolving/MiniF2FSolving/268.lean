import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find $9^{-1} \pmod{100}$, as a residue modulo 100.  (Give an answer between 0 and 99, inclusive.)

## Answer
89
-/

-- # Formal Answer
noncomputable abbrev Q268.gt := fun (answer : ℕ) => answer = (89 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q268
find (answer : ℕ) s.t.
: (answer < 100) ∧ ((9 * answer) % 100 = 1)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q268
find_all (answer : ℕ) iff
: (answer < 100) ∧ ((9 * answer) % 100 = 1)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
