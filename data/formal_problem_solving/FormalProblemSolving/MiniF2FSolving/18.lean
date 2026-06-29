import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the smallest positive integer that satisfies the congruence $30x \equiv 42 \pmod{47}$?

## Answer
39
-/

-- # Formal Answer
noncomputable abbrev Q18.gt := fun (answer : ℕ) => answer = (39 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q18
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {x : ℕ | 30 * x ≡ 42 [MOD 47]})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q18
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {x : ℕ | 30 * x ≡ 42 [MOD 47]})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
