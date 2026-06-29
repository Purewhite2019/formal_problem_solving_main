import FormalProblemSolving.Basic

/-
## Problem
How many positive integers $n$ are there such that $n$ is an exact divisor of at least one of the numbers $10^{40},20^{30}$?

## Answer
2301
-/

-- # Formal Answer
noncomputable abbrev Q80.gt := fun (answer : ℕ) => answer = (2301 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q80
find (answer : ℕ) s.t.
: ({n : ℤ | n > 0 ∧ (n ∣ 10 ^ 40 ∨ n ∣ 20 ^ 30)}.encard = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q80
find_all (answer : ℕ) iff
: ({n : ℤ | n > 0 ∧ (n ∣ 10 ^ 40 ∨ n ∣ 20 ^ 30)}.encard = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
