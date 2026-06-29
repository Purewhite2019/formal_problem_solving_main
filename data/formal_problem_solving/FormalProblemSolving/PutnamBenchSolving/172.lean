import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
Let $\{x\}$ denote the distance between the real number $x$ and the nearest integer.  For each positive integer $n$, evaluate \[F_n=\sum_{m=1}^{6n-1} \min(\{\frac{m}{6n}\},\{\frac{m}{3n}\}).\] (Here $\min(a,b)$ denotes the minimum of $a$ and $b$.)

## Answer
fun n => n
-/

-- # Formal Answer
noncomputable abbrev Q172.gt := fun (r : ℝ) (answer : ℕ → ℝ) => answer = fun (n : ℕ) => (↑n : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q172
  (r : ℝ)
find (answer : ℕ → ℝ) s.t.
  (dist_to_int : ℝ → ℝ)
  (h_dist_to_int : dist_to_int r = |r - round r|)
  (F : ℕ → ℝ)
  (hF : F = fun (n : ℕ) => ∑ m in Finset.Icc 1 (6 * n - 1), min (dist_to_int (m/(6*n)) ) (dist_to_int (m/(3*n))))
: (∀ n, n > 0 → F n = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q172
  (r : ℝ)
find_all (answer : ℕ → ℝ) iff
  (dist_to_int : ℝ → ℝ)
  (h_dist_to_int : dist_to_int r = |r - round r|)
  (F : ℕ → ℝ)
  (hF : F = fun (n : ℕ) => ∑ m in Finset.Icc 1 (6 * n - 1), min (dist_to_int (m/(6*n)) ) (dist_to_int (m/(3*n))))
: (∀ n, n > 0 → F n = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
