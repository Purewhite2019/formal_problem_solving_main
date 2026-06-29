import FormalProblemSolving.Basic
open Nat Set MeasureTheory Topology Filter

/-
## Problem
How many zeros does the function $f(x) = 2^x - 1 - x^2$ have on the real line?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q26.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q26
find (answer : ℕ) s.t.
  (f : ℝ → ℝ)
  (hf : f = fun x => 2^x - 1 - x^2)
: (answer = {x : ℝ | f x = 0}.ncard)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q26
find_all (answer : ℕ) iff
  (f : ℝ → ℝ)
  (hf : f = fun x => 2^x - 1 - x^2)
: (answer = {x : ℝ | f x = 0}.ncard)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
