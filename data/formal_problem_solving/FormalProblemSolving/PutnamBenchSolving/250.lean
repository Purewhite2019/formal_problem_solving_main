import FormalProblemSolving.Basic
open Topology Filter Nat

/-
## Problem
Let \( a_0 = \frac{5}{2} \) and \( a_k = a_{k-1}^2 - 2 \) for \( k \geq 1 \). Compute \( \prod_{k=0}^{\infty} \left(1 - \frac{1}{a_k}\right) \) in closed form.

## Answer
\( \frac{3}{7} \)
-/

-- # Formal Answer
noncomputable abbrev Q250.gt := fun (answer : ℝ) => answer = (3 / 7 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q250
find (answer : ℝ) s.t.
  (a : ℕ → ℝ)
  (a0 : a 0 = 5 / 2)
  (ak : ∀ k ≥ 1, a k = (a (k - 1)) ^ 2 - 2)
: (Tendsto (fun n : ℕ => ∏ k in Finset.range n, (1 - 1 / a k)) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q250
find_all (answer : ℝ) iff
  (a : ℕ → ℝ)
  (a0 : a 0 = 5 / 2)
  (ak : ∀ k ≥ 1, a k = (a (k - 1)) ^ 2 - 2)
: (Tendsto (fun n : ℕ => ∏ k in Finset.range n, (1 - 1 / a k)) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
