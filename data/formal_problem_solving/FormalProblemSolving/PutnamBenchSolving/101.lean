import FormalProblemSolving.Basic

/-
## Problem
Find, with explanation, the maximum value of $f(x)=x^3-3x$ on the set of all real numbers $x$ satisfying $x^4+36 \leq 13x^2$.

## Answer
18
-/

-- # Formal Answer
noncomputable abbrev Q101.gt := fun (answer : ℝ) => answer = (18 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q101
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (f : ℝ → ℝ)
  (hS : S = {x : ℝ | x ^ 4 + 36 ≤ 13 * x ^ 2})
  (hf : f = fun x ↦ x ^ 3 - 3 * x)
: (IsGreatest {f x | x ∈ S} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q101
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (f : ℝ → ℝ)
  (hS : S = {x : ℝ | x ^ 4 + 36 ≤ 13 * x ^ 2})
  (hf : f = fun x ↦ x ^ 3 - 3 * x)
: (IsGreatest {f x | x ∈ S} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
