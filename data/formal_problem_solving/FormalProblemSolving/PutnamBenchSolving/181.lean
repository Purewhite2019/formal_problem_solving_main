import FormalProblemSolving.Basic
open Filter Topology Metric

/-
## Problem
Let $A=\{(x,y):0\leq x,y<1\}$.  For $(x,y)\in A$, let \[S(x,y) = \sum_{\frac{1}{2}\leq \frac{m}{n}\leq 2} x^m y^n,\] where the sum ranges over all pairs $(m,n)$ of positive integers satisfying the indicated inequalities.  Evaluate \[\lim_{(x,y)\rightarrow (1,1), (x,y)\in A} (1-xy^2)(1-x^2y)S(x,y).\]

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q181.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q181
find (answer : ℝ) s.t.
  (A : Set (ℝ × ℝ))
  (hA : A = {xy | 0 ≤ xy.1 ∧ xy.1 < 1 ∧ 0 ≤ xy.2 ∧ xy.2 < 1})
  (S : ℝ → ℝ → ℝ)
  (hS : S = fun x y => ∑' m : ℕ, ∑' n : ℕ, if (m > 0 ∧ n > 0 ∧ 1/2 ≤ m/n ∧ m/n ≤ 2) then x^m * y^n else 0)
: (Tendsto (fun xy : (ℝ × ℝ) => (1 - xy.1 * xy.2^2) * (1 - xy.1^2 * xy.2) * (S xy.1 xy.2)) (𝓝[A] ⟨1,1⟩) (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q181
find_all (answer : ℝ) iff
  (A : Set (ℝ × ℝ))
  (hA : A = {xy | 0 ≤ xy.1 ∧ xy.1 < 1 ∧ 0 ≤ xy.2 ∧ xy.2 < 1})
  (S : ℝ → ℝ → ℝ)
  (hS : S = fun x y => ∑' m : ℕ, ∑' n : ℕ, if (m > 0 ∧ n > 0 ∧ 1/2 ≤ m/n ∧ m/n ≤ 2) then x^m * y^n else 0)
: (Tendsto (fun xy : (ℝ × ℝ) => (1 - xy.1 * xy.2^2) * (1 - xy.1^2 * xy.2) * (S xy.1 xy.2)) (𝓝[A] ⟨1,1⟩) (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
