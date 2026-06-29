import FormalProblemSolving.Basic
open Nat Filter Topology Real

/-
## Problem
Let $T$ be the triangle with vertices $(0, 0)$, $(a, 0)$, and $(0, a)$. Find $\lim_{a \to \infty} a^4 \exp(-a^3) \int_T \exp(x^3+y^3) \, dx \, dy$.

## Answer
2 / 9
-/

-- # Formal Answer
noncomputable abbrev Q82.gt := fun (answer : ℝ) => answer = (2 / 9 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q82
find (answer : ℝ) s.t.
  (F : ℝ → ℝ)
  (hF : F = fun a ↦ (a ^ 4 / exp (a ^ 3)) * ∫ x in (0)..a, ∫ y in (0)..(a - x), exp (x ^ 3 + y ^ 3))
: (Tendsto F atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q82
find_all (answer : ℝ) iff
  (F : ℝ → ℝ)
  (hF : F = fun a ↦ (a ^ 4 / exp (a ^ 3)) * ∫ x in (0)..a, ∫ y in (0)..(a - x), exp (x ^ 3 + y ^ 3))
: (Tendsto F atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
