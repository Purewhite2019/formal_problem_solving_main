import FormalProblemSolving.Basic
open Filter Polynomial Topology Nat

/-
## Problem
Is there an infinite sequence $a_0,a_1,a_2,\dots$ of nonzero real numbers such that for $n=1,2,3,\dots$ the polynomial $p_n(x)=a_0+a_1x+a_2x^2+\cdots+a_nx^n$ has exactly $n$ distinct real roots?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q137.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q137
find (answer : Prop) s.t.
: ((∃ a : ℕ → ℝ, (∀ i, a i ≠ 0) ∧
      (∀ n ≥ 1, (∑ i in Finset.Iic n, a i • X ^ i : Polynomial ℝ).roots.toFinset.card = n)) ↔
    answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q137
find_all (answer : Prop) iff
: ((∃ a : ℕ → ℝ, (∀ i, a i ≠ 0) ∧
      (∀ n ≥ 1, (∑ i in Finset.Iic n, a i • X ^ i : Polynomial ℝ).roots.toFinset.card = n)) ↔
    answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
