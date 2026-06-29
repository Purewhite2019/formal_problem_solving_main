import FormalProblemSolving.Basic
open Set Function Metric

/-
## Problem
Find necessary and sufficient conditions on positive integers $m$ and $n$ so that \[\sum_{i=0}^{mn-1} (-1)^{\lfloor i/m \rfloor +\lfloor i/n\rfloor}=0.\]

## Answer
{nm | let ⟨n,m⟩ := nm; multiplicity 2 n ≠ multiplicity 2 m}
-/

-- # Formal Answer
noncomputable abbrev Q177.gt := fun (n : ℕ) (m : ℕ) (answer : Set (ℕ × ℕ)) => answer = ({((n, m) : ℕ × ℕ) : ℕ × ℕ | (multiplicity (2 : ℕ) n : ℕ) ≠ (multiplicity (2 : ℕ) m : ℕ)} : Set (ℕ × ℕ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q177
  (n : ℕ)
  (m : ℕ)
find (answer : Set (ℕ × ℕ)) s.t.
  (quantity : ℕ → ℕ → ℤ)
  (hquantity : quantity = fun n m => ∑ i in Finset.range (m * n), (-1)^(i/m + i/n))
  (hnm : n > 0 ∧ m > 0)
: (quantity n m = 0 ↔ ⟨n, m⟩ ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q177
  (n : ℕ)
  (m : ℕ)
find_all (answer : Set (ℕ × ℕ)) iff
  (quantity : ℕ → ℕ → ℤ)
  (hquantity : quantity = fun n m => ∑ i in Finset.range (m * n), (-1)^(i/m + i/n))
  (hnm : n > 0 ∧ m > 0)
: (quantity n m = 0 ↔ ⟨n, m⟩ ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
