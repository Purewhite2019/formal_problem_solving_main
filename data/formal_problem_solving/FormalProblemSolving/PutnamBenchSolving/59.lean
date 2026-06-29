import FormalProblemSolving.Basic
open Set Topology Filter

/-
## Problem
If $0 < a < b$, find $$\lim_{t \to 0} \left( \int_{0}^{1}(bx + a(1-x))^t dx \right)^{\frac{1}{t}}$$ in terms of $a$ and $b$.

## Answer
fun (a, b) => (Real.exp (-1))*(b^b/a^a)^(1/(b-a))
-/

-- # Formal Answer
noncomputable abbrev Q59.gt := fun (answer : ℝ × ℝ → ℝ) => answer = fun (x : ℝ × ℝ) =>
  match x with
  | ((a, b) : ℝ × ℝ) =>
    ((Real.exp (-1 : ℝ) : ℝ) * (((b ^ b : ℝ) / (a ^ a : ℝ) : ℝ) ^ ((1 : ℝ) / (b - a : ℝ) : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q59
find (answer : ℝ × ℝ → ℝ) s.t.
: (∀ a b : ℝ, 0 < a ∧ a < b → Tendsto (fun t : ℝ => (∫ x in Icc 0 1, (b*x + a*(1 - x))^t)^(1/t)) (𝓝[≠] 0) (𝓝 (answer (a, b))))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q59
find_all (answer : ℝ × ℝ → ℝ) iff
: (∀ a b : ℝ, 0 < a ∧ a < b → Tendsto (fun t : ℝ => (∫ x in Icc 0 1, (b*x + a*(1 - x))^t)^(1/t)) (𝓝[≠] 0) (𝓝 (answer (a, b))))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
