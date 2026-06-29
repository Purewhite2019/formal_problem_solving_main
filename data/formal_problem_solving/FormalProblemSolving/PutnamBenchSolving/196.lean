import FormalProblemSolving.Basic
open Nat Topology Filter

/-
## Problem
Evaluate $\lim_{x \to 1^-} \prod_{n=0}^\infty \left(\frac{1+x^{n+1}}{1+x^n}\right)^{x^n}$.

## Answer
2 / e
-/

-- # Formal Answer
noncomputable abbrev Q196.gt := fun (answer : ℝ) => answer = ((2 : ℝ) / (Real.exp (1 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q196
find (answer : ℝ) s.t.
  (xprod : ℝ → ℝ)
  (hxprod : ∀ x ∈ Set.Ioo 0 1,
      Tendsto (fun N ↦ ∏ n in Finset.range N, ((1 + x ^ (n + 1)) / (1 + x ^ n)) ^ (x ^ n))
      atTop (𝓝 (xprod x)))
: (Tendsto xprod (𝓝[<] 1) (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q196
find_all (answer : ℝ) iff
  (xprod : ℝ → ℝ)
  (hxprod : ∀ x ∈ Set.Ioo 0 1,
      Tendsto (fun N ↦ ∏ n in Finset.range N, ((1 + x ^ (n + 1)) / (1 + x ^ n)) ^ (x ^ n))
      atTop (𝓝 (xprod x)))
: (Tendsto xprod (𝓝[<] 1) (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
