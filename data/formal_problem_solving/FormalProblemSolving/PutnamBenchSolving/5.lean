import FormalProblemSolving.Basic
open Topology Filter Polynomial

/-
## Problem
Let $S$ be the set of all numbers of the form $2^m3^n$, where $m$ and $n$ are integers, and let $P$ be the set of all positive real numbers. Is $S$ dense in $P$?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q5.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q5
find (answer : Prop) s.t.
  (S : Set ℝ)
  (hS : S = {2 ^ m * 3 ^ n | (m : ℤ) (n : ℤ)})
: (answer ↔ (closure S ⊇ Set.Ioi (0 : ℝ)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q5
find_all (answer : Prop) iff
  (S : Set ℝ)
  (hS : S = {2 ^ m * 3 ^ n | (m : ℤ) (n : ℤ)})
: (answer ↔ (closure S ⊇ Set.Ioi (0 : ℝ)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
