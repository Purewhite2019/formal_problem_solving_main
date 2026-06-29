import FormalProblemSolving.Basic
open EuclideanGeometry Filter Topology Set MeasureTheory Metric

/-
## Problem
Let $x : \mathbb{R} \to \mathbb{R}$ be a twice differentiable function whose second derivative is nonstrictly decreasing. If $x(t) - x(0) = s$, $x'(0) = 0$, and $x'(t) = v$ for some $t > 0$, find the maximum possible value of $t$ in terms of $s$ and $v$.

## Answer
$2s / v$
-/

-- # Formal Answer
noncomputable abbrev Q24.gt := fun (s : ℝ) (v : ℝ) (answer : ℝ → ℝ → ℝ) => answer = fun (s v : ℝ) => (((2 : ℝ) * s : ℝ) / v : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q24
  (s : ℝ)
  (v : ℝ)
find (answer : ℝ → ℝ → ℝ) s.t.
  (hs : s > 0)
  (hv : v > 0)
  (valid : ℝ → (ℝ → ℝ) → Prop)
  (hvalid : ∀ t x, valid t x ↔
      DifferentiableOn ℝ x (Set.Icc 0 t) ∧ DifferentiableOn ℝ (deriv x) (Set.Icc 0 t) ∧
      AntitoneOn (deriv (deriv x)) (Set.Icc 0 t) ∧
      deriv x 0 = 0 ∧ deriv x t = v ∧ x t - x 0 = s)
: (IsGreatest {t | ∃ x : ℝ → ℝ, valid t x} (answer s v))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q24
  (s : ℝ)
  (v : ℝ)
find_all (answer : ℝ → ℝ → ℝ) iff
  (hs : s > 0)
  (hv : v > 0)
  (valid : ℝ → (ℝ → ℝ) → Prop)
  (hvalid : ∀ t x, valid t x ↔
      DifferentiableOn ℝ x (Set.Icc 0 t) ∧ DifferentiableOn ℝ (deriv x) (Set.Icc 0 t) ∧
      AntitoneOn (deriv (deriv x)) (Set.Icc 0 t) ∧
      deriv x 0 = 0 ∧ deriv x t = v ∧ x t - x 0 = s)
: (IsGreatest {t | ∃ x : ℝ → ℝ, valid t x} (answer s v))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
