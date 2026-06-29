import FormalProblemSolving.Basic
open Filter Topology Set Nat

/-
## Problem
Let $F_0(x)=\ln x$. For $n \geq 0$ and $x>0$, let $F_{n+1}(x)=\int_0^x F_n(t)\,dt$. Evaluate $\lim_{n \to \infty} \frac{n!F_n(1)}{\ln n}$.

## Answer
-1
-/

-- # Formal Answer
noncomputable abbrev Q218.gt := fun (answer : ℝ) => answer = (-1 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q218
find (answer : ℝ) s.t.
  (F : ℕ → ℝ → ℝ)
  (hF0 : ∀ x : ℝ, F 0 x = Real.log x)
  (hFn : ∀ n : ℕ, ∀ x > 0, F (n + 1) x = ∫ t in Set.Ioo 0 x, F n t)
: (Tendsto (fun n : ℕ => ((n)! * F n 1) / Real.log n) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q218
find_all (answer : ℝ) iff
  (F : ℕ → ℝ → ℝ)
  (hF0 : ∀ x : ℝ, F 0 x = Real.log x)
  (hFn : ∀ n : ℕ, ∀ x > 0, F (n + 1) x = ∫ t in Set.Ioo 0 x, F n t)
: (Tendsto (fun n : ℕ => ((n)! * F n 1) / Real.log n) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
