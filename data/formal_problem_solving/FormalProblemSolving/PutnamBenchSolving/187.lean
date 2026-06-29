import FormalProblemSolving.Basic
open Topology Filter Polynomial Set

/-
## Problem
Assume that $(a_n)_{n \geq 1}$ is an increasing sequence of positive real numbers such that $\lim a_n/n=0$. Must there exist infinitely many positive integers $n$ such that $a_{n-i}+a_{n+i}<2a_n$ for $i=1,2,\ldots,n-1$?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q187.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q187
find (answer : Prop) s.t.
  (a : ℤ → ℝ)
  (h_pos_inc : ∀ n ≥ 1, a n > 0 ∧ a n < a (n + 1))
  (h_limit : Tendsto (fun n : ℤ => a (n + 1) / (n + 1)) atTop (𝓝 0))
: ({n : ℤ | n > 0 ∧ (∀ i ∈ Set.Icc 1 (n - 1), a (n - i) + a (n + i) < 2 * a n)}.Infinite ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q187
find_all (answer : Prop) iff
  (a : ℤ → ℝ)
  (h_pos_inc : ∀ n ≥ 1, a n > 0 ∧ a n < a (n + 1))
  (h_limit : Tendsto (fun n : ℤ => a (n + 1) / (n + 1)) atTop (𝓝 0))
: ({n : ℤ | n > 0 ∧ (∀ i ∈ Set.Icc 1 (n - 1), a (n - i) + a (n + i) < 2 * a n)}.Infinite ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
