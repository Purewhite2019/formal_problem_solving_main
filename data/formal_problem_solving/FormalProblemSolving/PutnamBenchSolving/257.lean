import FormalProblemSolving.Basic

/-
## Problem
For each real number $x$, let
\[
f(x) = \sum_{n\in S_x} \frac{1}{2^n},
\]
where $S_x$ is the set of positive integers $n$ for which $\lfloor nx \rfloor$ is even. What is the largest real number $L$ such that $f(x) \geq L$ for all $x \in [0,1)$? (As usual, $\lfloor z \rfloor$ denotes the greatest integer less than or equal to $z$.)

## Answer
4/7
-/

-- # Formal Answer
noncomputable abbrev Q257.gt := fun (answer : ℝ) => answer = (4 / 7 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q257
find (answer : ℝ) s.t.
  (S : ℝ → Set ℤ)
  (f : ℝ → ℝ)
  (hS : S = fun (x : ℝ) ↦ {n : ℤ | n > 0 ∧ Even ⌊n * x⌋})
  (hf : f = fun (x : ℝ) ↦ ∑' n : S x, 1 / 2 ^ (n : ℤ))
: (IsGreatest {l : ℝ | ∀ x ∈ Set.Ico 0 1, f x ≥ l} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q257
find_all (answer : ℝ) iff
  (S : ℝ → Set ℤ)
  (f : ℝ → ℝ)
  (hS : S = fun (x : ℝ) ↦ {n : ℤ | n > 0 ∧ Even ⌊n * x⌋})
  (hf : f = fun (x : ℝ) ↦ ∑' n : S x, 1 / 2 ^ (n : ℤ))
: (IsGreatest {l : ℝ | ∀ x ∈ Set.Ico 0 1, f x ≥ l} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
