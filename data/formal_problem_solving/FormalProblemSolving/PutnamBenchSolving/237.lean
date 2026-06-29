import FormalProblemSolving.Basic
open Topology Filter

/-
## Problem
Let $a_1,a_2,\dots$ and $b_1,b_2,\dots$ be sequences of positive real numbers such that $a_1 = b_1 = 1$ and $b_n = b_{n-1} a_n - 2$ for$n=2,3,\dots$. Assume that the sequence $(b_j)$ is bounded. Prove tha \[ S = \sum_{n=1}^\infty \frac{1}{a_1...a_n} \] converges, and evaluate $S$.

## Answer
3/2
-/

-- # Formal Answer
noncomputable abbrev Q237.gt := fun (answer : ℝ) => answer = (3 / 2 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q237
find (answer : ℝ) s.t.
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (habn : ∀ n : ℕ, a n > 0 ∧ b n > 0)
  (hab1 : a 0 = 1 ∧ b 0 = 1)
  (hb : ∀ n ≥ 1, b n = b (n-1) * a n - 2)
  (hbnd : ∃ B : ℝ, ∀ n : ℕ, |b n| ≤ B)
: (Tendsto (fun n => ∑ i : Fin n, 1/(∏ j : Fin (i + 1), (a j))) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q237
find_all (answer : ℝ) iff
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (habn : ∀ n : ℕ, a n > 0 ∧ b n > 0)
  (hab1 : a 0 = 1 ∧ b 0 = 1)
  (hb : ∀ n ≥ 1, b n = b (n-1) * a n - 2)
  (hbnd : ∃ B : ℝ, ∀ n : ℕ, |b n| ≤ B)
: (Tendsto (fun n => ∑ i : Fin n, 1/(∏ j : Fin (i + 1), (a j))) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
