import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve the congruence $5n \equiv 8 \pmod{17}$, as a residue modulo 17.  (Give an answer between 0 and 16.)

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q175.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q175
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : (5 * n) % 17 = 8)
: (answer = n % 17)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q175
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : (5 * n) % 17 = 8)
: (answer = n % 17)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
