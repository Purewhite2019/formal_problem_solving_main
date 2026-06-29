import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Define
\[p = \sum_{k = 1}^\infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]Find a way to write
\[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}\]in terms of $p$ and $q.$

## Answer
p - q
-/

-- # Formal Answer
noncomputable abbrev Q351.gt := fun (p : ℝ) (q : ℝ) (answer : ℝ) => answer = (p - q : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q351
  (p : ℝ)
  (q : ℝ)
find (answer : ℝ) s.t.
  (h_p : p = ∑' (k : ℕ), 1 / ((k : ℝ) + 1)^2)
  (h_q : q = ∑' (k : ℕ), 1 / ((k : ℝ) + 1)^3)
: (answer = ∑' (j : ℕ), ∑' (k : ℕ), 1 / ((j : ℝ) + (k : ℝ) + 2)^3)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q351
  (p : ℝ)
  (q : ℝ)
find_all (answer : ℝ) iff
  (h_p : p = ∑' (k : ℕ), 1 / ((k : ℝ) + 1)^2)
  (h_q : q = ∑' (k : ℕ), 1 / ((k : ℝ) + 1)^3)
: (answer = ∑' (j : ℕ), ∑' (k : ℕ), 1 / ((j : ℝ) + (k : ℝ) + 2)^3)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
