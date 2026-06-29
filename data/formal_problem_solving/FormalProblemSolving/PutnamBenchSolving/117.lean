import FormalProblemSolving.Basic
open Set

/-
## Problem
A not uncommon calculus mistake is to believe that the product rule for derivatives says that $(fg)' = f'g'$. If $f(x)=e^{x^2}$, determine, with proof, whether there exists an open interval $(a,b)$ and a nonzero function $g$ defined on $(a,b)$ such that this wrong product rule is true for $x$ in $(a,b)$.

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q117.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q117
find (answer : Prop) s.t.
  (f : ℝ → ℝ)
  (hf : f = fun x ↦ Real.exp (x ^ 2))
: (answer ↔
      (∃ a b : ℝ,
        a < b ∧
        ∃ g : ℝ → ℝ,
          (∃ x ∈ Ioo a b, g x ≠ 0) ∧
          DifferentiableOn ℝ g (Ioo a b) ∧
          ∀ x ∈ Ioo a b, deriv (fun y ↦ f y * g y) x = (deriv f x) * (deriv g x)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q117
find_all (answer : Prop) iff
  (f : ℝ → ℝ)
  (hf : f = fun x ↦ Real.exp (x ^ 2))
: (answer ↔
      (∃ a b : ℝ,
        a < b ∧
        ∃ g : ℝ → ℝ,
          (∃ x ∈ Ioo a b, g x ≠ 0) ∧
          DifferentiableOn ℝ g (Ioo a b) ∧
          ∀ x ∈ Ioo a b, deriv (fun y ↦ f y * g y) x = (deriv f x) * (deriv g x)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
