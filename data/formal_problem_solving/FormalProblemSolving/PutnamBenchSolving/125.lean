import FormalProblemSolving.Basic
open Real

/-
## Problem
Evaluate $\int_0^a \int_0^b e^{\max\{b^2x^2,a^2y^2\}}\,dy\,dx$ where $a$ and $b$ are positive.

## Answer
$\frac{e^{a^2 b^2} - 1}{a b}$
-/

-- # Formal Answer
noncomputable abbrev Q125.gt := fun (a : ℝ) (b : ℝ) (answer : ℝ → ℝ → ℝ) => answer = fun (a b : ℝ) => (((rexp ((a ^ (2 : ℕ) : ℝ) * (b ^ (2 : ℕ) : ℝ) : ℝ) : ℝ) - (1 : ℝ) : ℝ) / (a * b : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q125
  (a : ℝ)
  (b : ℝ)
find (answer : ℝ → ℝ → ℝ) s.t.
  (abpos : a > 0 ∧ b > 0)
: (∫ x in Set.Ioo 0 a, ∫ y in Set.Ioo 0 b, Real.exp (max (b ^ 2 * x ^ 2) (a ^ 2 * y ^ 2)) = answer a b)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q125
  (a : ℝ)
  (b : ℝ)
find_all (answer : ℝ → ℝ → ℝ) iff
  (abpos : a > 0 ∧ b > 0)
: (∫ x in Set.Ioo 0 a, ∫ y in Set.Ioo 0 b, Real.exp (max (b ^ 2 * x ^ 2) (a ^ 2 * y ^ 2)) = answer a b)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
