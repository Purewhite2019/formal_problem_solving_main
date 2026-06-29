import FormalProblemSolving.Basic

/-
## Problem
Find the number of positive integers $N$ satisfying: (i) $N$ is divisible by $2020$, (ii) $N$ has at most $2020$ decimal digits, (iii) The decimal digits of $N$ are a string of consecutive ones followed by a string of consecutive zeros.

## Answer
508536
-/

-- # Formal Answer
noncomputable abbrev Q290.gt := fun (answer : ℕ) => answer = (508536 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q290
find (answer : ℕ) s.t.
: (Set.ncard {x : ℕ | (2020 ∣ x) ∧ (Nat.log 10 x) + 1 ≤ 2020 ∧ (∃ k l, k ≥ l ∧ x = ∑ i in Finset.range (k - l + 1), 10 ^ (i + l))} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q290
find_all (answer : ℕ) iff
: (Set.ncard {x : ℕ | (2020 ∣ x) ∧ (Nat.log 10 x) + 1 ≤ 2020 ∧ (∃ k l, k ≥ l ∧ x = ∑ i in Finset.range (k - l + 1), 10 ^ (i + l))} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
