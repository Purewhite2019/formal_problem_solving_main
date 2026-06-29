import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If
\[f(n + 1) = (-1)^{n + 1} n - 2f(n)\]for $n \ge 1,$ and $f(1) = f(1986),$ compute
\[f(1) + f(2) + f(3) + \dots + f(1985).\]

## Answer
331
-/

-- # Formal Answer
noncomputable abbrev Q307.gt := fun (answer : ℝ) => answer = (331 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q307
find (answer : ℝ) s.t.
  (f : ℕ → ℝ)
  (h_rec : ∀ n ≥ 1, f (n + 1) = (-1) ^ (n + 1) * n - 2 * f n)
  (h_periodic : f 1 = f 1986)
: (answer = ∑ n in Finset.Icc 1 1985, f n)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q307
find_all (answer : ℝ) iff
  (f : ℕ → ℝ)
  (h_rec : ∀ n ≥ 1, f (n + 1) = (-1) ^ (n + 1) * n - 2 * f n)
  (h_periodic : f 1 = f 1986)
: (answer = ∑ n in Finset.Icc 1 1985, f n)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
