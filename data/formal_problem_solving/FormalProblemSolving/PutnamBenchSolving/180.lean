import FormalProblemSolving.Basic
open Filter Topology Metric

/-
## Problem
Sum the series \[\sum_{m=1}^\infty \sum_{n=1}^\infty \frac{m^2 n}{3^m(n3^m+m3^n)}.\]

## Answer
9/32
-/

-- # Formal Answer
noncomputable abbrev Q180.gt := fun (answer : ℝ) => answer = (9 / 32 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q180
find (answer : ℝ) s.t.
: (Tendsto (fun i => ∑ m in Finset.range i, ∑' n : ℕ, (((m + 1)^2*(n+1))/(3^(m + 1) * ((n+1)*3^(m + 1) + (m + 1)*3^(n+1))) : ℝ)) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q180
find_all (answer : ℝ) iff
: (Tendsto (fun i => ∑ m in Finset.range i, ∑' n : ℕ, (((m + 1)^2*(n+1))/(3^(m + 1) * ((n+1)*3^(m + 1) + (m + 1)*3^(n+1))) : ℝ)) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
