import FormalProblemSolving.Basic
open Nat Set Topology Filter

/-
## Problem
Find an integral formula (i.e., a function $z$ such that $y(x) = \int_{1}^{x} z(t) dt$) for the solution of the differential equation $$\delta (\delta - 1) (\delta - 2) \cdots (\delta - n + 1) y = f(x)$$ with the initial conditions $y(1) = y'(1) = \cdots = y^{(n-1)}(1) = 0$, where $n \in \mathbb{N}$, $f$ is continuous for all $x \ge 1$, and $\delta$ denotes $x\frac{d}{dx}$.

## Answer
$(x - t)^{n-1} \cdot f(t) / ((n-1)! \cdot t^n)$
-/

-- # Formal Answer
noncomputable abbrev Q3.gt := fun (n : ℕ) (f : ℝ → ℝ) (answer : (ℝ → ℝ) → ℕ → ℝ → ℝ → ℝ) => answer = fun (f : ℝ → ℝ) (n : ℕ) (x t : ℝ) =>
  ((((x - t : ℝ) ^ (n - (1 : ℕ) : ℕ) : ℝ) * (f t : ℝ) : ℝ) / ((↑((n - (1 : ℕ) : ℕ)! : ℕ) : ℝ) * (t ^ n : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q3
  (n : ℕ)
  (f : ℝ → ℝ)
find (answer : (ℝ → ℝ) → ℕ → ℝ → ℝ → ℝ) s.t.
  (P : ℕ → (ℝ → ℝ) → (ℝ → ℝ))
  (hP : P 0 = id ∧ ∀ i y, P (i + 1) y = P i (fun x ↦ x * deriv y x - i * y x))
  (hn : 0 < n)
  (y : ℝ → ℝ)
  (hf : ContinuousOn f (Ici 1))
  (hy : ContDiffOn ℝ n y (Ici 1))
: ((∀ i < n, deriv^[i] y 1 = 0) ∧ (Ici 1).EqOn (P n y) f ↔
    ∀ x ≥ 1, y x = ∫ t in (1 : ℝ)..x, answer f n x t)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q3
  (n : ℕ)
  (f : ℝ → ℝ)
find_all (answer : (ℝ → ℝ) → ℕ → ℝ → ℝ → ℝ) iff
  (P : ℕ → (ℝ → ℝ) → (ℝ → ℝ))
  (hP : P 0 = id ∧ ∀ i y, P (i + 1) y = P i (fun x ↦ x * deriv y x - i * y x))
  (hn : 0 < n)
  (y : ℝ → ℝ)
  (hf : ContinuousOn f (Ici 1))
  (hy : ContDiffOn ℝ n y (Ici 1))
: ((∀ i < n, deriv^[i] y 1 = 0) ∧ (Ici 1).EqOn (P n y) f ↔
    ∀ x ≥ 1, y x = ∫ t in (1 : ℝ)..x, answer f n x t)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
