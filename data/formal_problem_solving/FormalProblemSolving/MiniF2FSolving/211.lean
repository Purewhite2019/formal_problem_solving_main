import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The greatest common divisor of positive integers $m$ and $n$ is 6. The least common multiple of $m$ and $n$ is 126. What is the least possible value of $m+n$?

## Answer
60
-/

-- # Formal Answer
noncomputable abbrev Q211.gt := fun (answer : ℕ) => answer = (60 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q211
find (answer : ℕ) s.t.
: (IsLeast {s : ℕ | ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ Nat.gcd m n = 6 ∧ Nat.lcm m n = 126 ∧ s = m + n} answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q211
find_all (answer : ℕ) iff
: (IsLeast {s : ℕ | ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ Nat.gcd m n = 6 ∧ Nat.lcm m n = 126 ∧ s = m + n} answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
