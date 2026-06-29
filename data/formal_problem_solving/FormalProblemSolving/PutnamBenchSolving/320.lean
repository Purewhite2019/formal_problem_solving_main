import FormalProblemSolving.Basic
open Nat

/-
## Problem
For each positive integer $n$, let $k(n)$ be the number of ones in the binary representation of $2023 * n$. What is the minimum value of $k(n)$?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q320.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q320
find (answer : ℕ) s.t.
: (sInf {(digits 2 (2023 * n)).sum | n > 0} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q320
find_all (answer : ℕ) iff
: (sInf {(digits 2 (2023 * n)).sum | n > 0} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
