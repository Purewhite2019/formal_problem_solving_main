import FormalProblemSolving.Basic
open Topology Filter Nat

/-
## Problem
Suppose that \( f \) is a function on the interval \([1,3]\) such that \(-1 \leq f(x) \leq 1\) for all \( x \) and \( \int_{1}^{3} f(x) \, dx = 0 \). How large can \(\int_{1}^{3} \frac{f(x)}{x} \, dx \) be?

## Answer
\( \log(4/3) \)
-/

-- # Formal Answer
noncomputable abbrev Q254.gt := fun (answer : ℝ) => answer = (Real.log (4 / 3 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q254
find (answer : ℝ) s.t.
: (IsGreatest {t | ∃ f : ℝ → ℝ,
    (∀ x : Set.Icc (1 : ℝ) 3, -1 ≤ f x ∧ f x ≤ 1) ∧
    (∫ x in Set.Ioo 1 3, f x = 0) ∧
    (∫ x in Set.Ioo 1 3, (f x) / x) = t}
  answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q254
find_all (answer : ℝ) iff
: (IsGreatest {t | ∃ f : ℝ → ℝ,
    (∀ x : Set.Icc (1 : ℝ) 3, -1 ≤ f x ∧ f x ≤ 1) ∧
    (∫ x in Set.Ioo 1 3, f x = 0) ∧
    (∫ x in Set.Ioo 1 3, (f x) / x) = t}
  answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
