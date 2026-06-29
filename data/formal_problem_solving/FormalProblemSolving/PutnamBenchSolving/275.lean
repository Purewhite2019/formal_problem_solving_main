import FormalProblemSolving.Basic
open Topology Filter Real Function Nat

/-
## Problem
Find the number of ordered $64$-tuples $(x_0,x_1,\dots,x_{63})$ such that $x_0,x_1,\dots,x_{63}$ are distinct elements of $\{1,2,\dots,2017\}$ and
\[
x_0 + x_1 + 2x_2 + 3x_3 + \cdots + 63 x_{63}
\]
is divisible by 2017.

## Answer
2016! / 1953! - 63! * 2016
-/

-- # Formal Answer
noncomputable abbrev Q275.gt := fun (answer : ℕ) => answer = ((((2016 : ℕ)! : ℕ) / ((1953 : ℕ)! : ℕ) : ℕ) - (((63 : ℕ)! : ℕ) * (2016 : ℕ) : ℕ) : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q275
find (answer : ℕ) s.t.
  (S : Finset (Finset.range 64 → Finset.Icc 1 2017))
  (hs : ∀ x, x ∈ S ↔ (Injective x ∧ (2017 ∣ (∑ i : Finset.range 64, if i ≤ (⟨1, by norm_num⟩ : Finset.range 64) then (x i : ℤ) else i * (x i : ℤ)))))
: (S.card = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q275
find_all (answer : ℕ) iff
  (S : Finset (Finset.range 64 → Finset.Icc 1 2017))
  (hs : ∀ x, x ∈ S ↔ (Injective x ∧ (2017 ∣ (∑ i : Finset.range 64, if i ≤ (⟨1, by norm_num⟩ : Finset.range 64) then (x i : ℤ) else i * (x i : ℤ)))))
: (S.card = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
