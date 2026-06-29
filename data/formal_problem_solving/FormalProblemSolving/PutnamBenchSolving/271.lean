import FormalProblemSolving.Basic
open Polynomial Filter Topology Real Set Nat List

/-
## Problem
Evaluate $\sum_{k=1}^\infty \frac{(-1)^{k-1}}{k} \sum_{n=0}^\infty \frac{1}{k2^n+1}$.

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q271.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q271
find (answer : ℝ) s.t.
: (∑' k : ℕ, ((-1 : ℝ) ^ ((k + 1 : ℤ) - 1) / (k + 1 : ℝ)) * ∑' n : ℕ, (1 : ℝ) / ((k + 1) * (2 ^ n) + 1) = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q271
find_all (answer : ℝ) iff
: (∑' k : ℕ, ((-1 : ℝ) ^ ((k + 1 : ℤ) - 1) / (k + 1 : ℝ)) * ∑' n : ℕ, (1 : ℝ) / ((k + 1) * (2 ^ n) + 1) = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
