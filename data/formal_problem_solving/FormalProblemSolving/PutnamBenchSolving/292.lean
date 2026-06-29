import FormalProblemSolving.Basic
open Filter Topology Set

/-
## Problem
Let $a_0 = \pi/2$, and let $a_n = \sin(a_{n-1})$ for $n \geq 1$. Determine whether
\[
\sum_{n=1}^\infty a_n^2
\]
converges.

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q292.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q292
find (answer : Prop) s.t.
  (a : ℕ → ℝ)
  (ha0 : a 0 = Real.pi / 2)
  (ha : ∀ n, a (n+1) = Real.sin (a n))
: ((∃ L, Tendsto (fun m : ℕ => ∑ n in Finset.Icc 1 m, (a n)^2) atTop (𝓝 L)) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q292
find_all (answer : Prop) iff
  (a : ℕ → ℝ)
  (ha0 : a 0 = Real.pi / 2)
  (ha : ∀ n, a (n+1) = Real.sin (a n))
: ((∃ L, Tendsto (fun m : ℕ => ∑ n in Finset.Icc 1 m, (a n)^2) atTop (𝓝 L)) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
