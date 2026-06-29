import FormalProblemSolving.Basic
open Set Real

/-
## Problem
Find
\[
\sum_{i=1}^{\infty} \sum_{j=1}^{\infty} \frac{1}{i^2j + 2ij + ij^2}.
\]

## Answer
7 / 4
-/

-- # Formal Answer
noncomputable abbrev Q53.gt := fun (answer : ℚ) => answer = (7 / 4 : ℚ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q53
find (answer : ℚ) s.t.
: ((∑' i : ℕ+, ∑' j : ℕ+, (1 : ℚ) / (i ^ 2 * j + 2 * i * j + i * j ^ 2)) = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q53
find_all (answer : ℚ) iff
: ((∑' i : ℕ+, ∑' j : ℕ+, (1 : ℚ) / (i ^ 2 * j + 2 * i * j + i * j ^ 2)) = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
