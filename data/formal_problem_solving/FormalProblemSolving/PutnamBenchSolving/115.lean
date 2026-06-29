import FormalProblemSolving.Basic
open MvPolynomial Real Nat Filter Topology

/-
## Problem
Let $(x_1,y_1) = (0.8, 0.6)$ and let $x_{n+1} = x_n \cos y_n - y_n \sin y_n$ and $y_{n+1}= x_n \sin y_n + y_n \cos y_n$ for $n=1,2,3,\dots$. For each of $\lim_{n\to \infty} x_n$ and $\lim_{n \to \infty} y_n$, prove that the limit exists and find it or prove that the limit does not exist.

## Answer
(True, -1, True, 0)
-/

-- # Formal Answer
noncomputable abbrev Q115.gt := fun (answer : Prop × ℝ × Prop × ℝ) => answer = ((True, (((-1 : ℝ), ((True, (0 : ℝ)) : Prop × ℝ)) : ℝ × Prop × ℝ)) : Prop × ℝ × Prop × ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q115
find (answer : Prop × ℝ × Prop × ℝ) s.t.
  (x : ℕ → ℝ)
  (y : ℕ → ℝ)
  (hxy1 : (x 1, y 1) = (0.8, 0.6))
  (hx : ∀ n ≥ 1, x (n + 1) = (x n) * cos (y n) - (y n) * sin (y n))
  (hy : ∀ n ≥ 1, y (n + 1) = (x n) * sin (y n) + (y n) * cos (y n))
: (let (existsx, limx, existsy, limy) := answer
((∃ c : ℝ, Tendsto x atTop (𝓝 c)) → existsx) ∧
(existsx → Tendsto x atTop (𝓝 limx)) ∧
((∃ c : ℝ, Tendsto y atTop (𝓝 c)) → existsy) ∧
(existsy → Tendsto y atTop (𝓝 limy)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q115
find_all (answer : Prop × ℝ × Prop × ℝ) iff
  (x : ℕ → ℝ)
  (y : ℕ → ℝ)
  (hxy1 : (x 1, y 1) = (0.8, 0.6))
  (hx : ∀ n ≥ 1, x (n + 1) = (x n) * cos (y n) - (y n) * sin (y n))
  (hy : ∀ n ≥ 1, y (n + 1) = (x n) * sin (y n) + (y n) * cos (y n))
: (let (existsx, limx, existsy, limy) := answer
((∃ c : ℝ, Tendsto x atTop (𝓝 c)) → existsx) ∧
(existsx → Tendsto x atTop (𝓝 limx)) ∧
((∃ c : ℝ, Tendsto y atTop (𝓝 c)) → existsy) ∧
(existsy → Tendsto y atTop (𝓝 limy)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
