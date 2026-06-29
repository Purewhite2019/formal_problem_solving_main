import FormalProblemSolving.Basic
open Topology Filter Nat Function

/-
## Problem
Let $f$ be an infinitely differentiable real-valued function defined on the real numbers. If
\[
f\left( \frac{1}{n} \right) = \frac{n^2}{n^2 + 1}, \qquad n = 1, 2, 3, \dots,
\]
compute the values of the derivatives $f^{(k)}(0), k = 1, 2, 3, \dots$.

## Answer
fun k ↦ ite (Even k) ((-1) ^ (k / 2) * factorial k) 0
-/

-- # Formal Answer
noncomputable abbrev Q147.gt := fun (answer : ℕ → ℝ) => answer = fun (k : ℕ) => (if Even k then (((-1 : ℝ) ^ (k / (2 : ℕ) : ℕ) : ℝ) * (↑(k ! : ℕ) : ℝ) : ℝ) else (0 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q147
find (answer : ℕ → ℝ) s.t.
  (f : ℝ → ℝ)
  (hfdiff : ContDiff ℝ ⊤ f)
  (hf : ∀ n : ℕ, n > 0 → f (1 / n) = n ^ 2 / (n ^ 2 + 1))
: (∀ k : ℕ, k > 0 → iteratedDeriv k f 0 = answer k)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q147
find_all (answer : ℕ → ℝ) iff
  (f : ℝ → ℝ)
  (hfdiff : ContDiff ℝ ⊤ f)
  (hf : ∀ n : ℕ, n > 0 → f (1 / n) = n ^ 2 / (n ^ 2 + 1))
: (∀ k : ℕ, k > 0 → iteratedDeriv k f 0 = answer k)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
