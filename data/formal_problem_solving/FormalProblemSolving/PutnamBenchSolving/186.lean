import FormalProblemSolving.Basic
open Topology Filter Polynomial Set

/-
## Problem
For any positive integer $n$, let $\langle n \rangle$ denote the closest integer to $\sqrt{n}$. Evaluate $\sum_{n=1}^\infty \frac{2^{\langle n \rangle}+2^{-\langle n \rangle}}{2^n}$.

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q186.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q186
find (answer : ℝ) s.t.
: (∑' n : Set.Ici 1, ((2 : ℝ) ^ (round (Real.sqrt n)) + (2 : ℝ) ^ (-round (Real.sqrt n))) / 2 ^ (n : ℝ) = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q186
find_all (answer : ℝ) iff
: (∑' n : Set.Ici 1, ((2 : ℝ) ^ (round (Real.sqrt n)) + (2 : ℝ) ^ (-round (Real.sqrt n))) / 2 ^ (n : ℝ) = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
