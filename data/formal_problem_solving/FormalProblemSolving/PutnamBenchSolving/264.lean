import FormalProblemSolving.Basic
open Polynomial Filter Topology Real Set Nat

/-
## Problem
Given a positive integer $n$, let $M(n)$ be the largest integer $m$ such that
\[
\binom{m}{n-1} > \binom{m-1}{n}.
\]
Evaluate
\[
\lim_{n \to \infty} \frac{M(n)}{n}.
\]

## Answer
(3 + √5) / 2
-/

-- # Formal Answer
noncomputable abbrev Q264.gt := fun (answer : ℝ) => answer = (((3 : ℝ) + (√(5 : ℝ) : ℝ) : ℝ) / (2 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q264
find (answer : ℝ) s.t.
  (M : ℕ → ℕ)
  (hM : ∀ n > 0, IsGreatest {m | 0 < m ∧ (m - 1).choose n < m.choose (n - 1)} (M n))
: (Tendsto (fun n ↦ M n / (n : ℝ)) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q264
find_all (answer : ℝ) iff
  (M : ℕ → ℕ)
  (hM : ∀ n > 0, IsGreatest {m | 0 < m ∧ (m - 1).choose n < m.choose (n - 1)} (M n))
: (Tendsto (fun n ↦ M n / (n : ℝ)) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
