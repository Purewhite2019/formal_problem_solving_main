import FormalProblemSolving.Basic
open Filter Topology Metric

/-
## Problem
For an integer $n\geq 3$, let $\theta=2\pi/n$.  Evaluate the determinant of the $n\times n$ matrix $I+A$, where $I$ is the $n\times n$ identity matrix and $A=(a_{jk})$ has entries $a_{jk}=\cos(j\theta+k\theta)$ for all $j,k$.

## Answer
fun n => 1 - n^2/4
-/

-- # Formal Answer
noncomputable abbrev Q182.gt := fun (n : ℕ) (answer : ℕ → ℝ) => answer = fun (n : ℕ) => ((1 : ℝ) - (((↑n : ℝ) ^ (2 : ℕ) : ℝ) / (4 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q182
  (n : ℕ)
find (answer : ℕ → ℝ) s.t.
  (hn : n ≥ 3)
  (theta : ℝ)
  (htheta : theta = 2 * Real.pi / n)
  (A : Matrix (Fin n) (Fin n) ℝ)
  (hA : A = fun j k => Real.cos ((j.1 + 1) * theta + (k.1 + 1) * theta))
: ((1 + A).det = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q182
  (n : ℕ)
find_all (answer : ℕ → ℝ) iff
  (hn : n ≥ 3)
  (theta : ℝ)
  (htheta : theta = 2 * Real.pi / n)
  (A : Matrix (Fin n) (Fin n) ℝ)
  (hA : A = fun j k => Real.cos ((j.1 + 1) * theta + (k.1 + 1) * theta))
: ((1 + A).det = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
