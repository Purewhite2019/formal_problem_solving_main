import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The greatest common divisor of two integers is $(x+3)$ and their least common multiple is $x(x+3)$, where $x$ is a positive integer. If one of the integers is 40, what is the smallest possible value of the other one?

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q153.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q153
find (answer : ℕ) s.t.
: (IsLeast { a : ℕ | 0 < a ∧ ∃ (x : ℕ), 0 < x ∧ Nat.gcd a 40 = x + 3 ∧ Nat.lcm a 40 = x * (x + 3) } answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q153
find_all (answer : ℕ) iff
: (IsLeast { a : ℕ | 0 < a ∧ ∃ (x : ℕ), 0 < x ∧ Nat.gcd a 40 = x + 3 ∧ Nat.lcm a 40 = x * (x + 3) } answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
