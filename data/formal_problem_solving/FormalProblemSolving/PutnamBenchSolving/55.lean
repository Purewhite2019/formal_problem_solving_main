import FormalProblemSolving.Basic

/-
## Problem
For which positive integers $n$ and $a_1, a_2, \dots, a_n$ with $\sum_{i = 1}^{n} a_i = 1979$ does $\prod_{i = 1}^{n} a_i$ attain the greatest value?

## Answer
Multiset.replicate 659 3 + {2}
-/

-- # Formal Answer
noncomputable abbrev Q55.gt := fun (answer : Multiset ℕ) => answer = ((Multiset.replicate (659 : ℕ) (3 : ℕ) : Multiset ℕ) + ({(2 : ℕ)} : Multiset ℕ) : Multiset ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q55
find (answer : Multiset ℕ) s.t.
  (P : Multiset ℕ → Prop)
  (hP : ∀ a, P a ↔ Multiset.card a > 0 ∧ (∀ i ∈ a, i > 0) ∧ a.sum = 1979)
: (P answer ∧ ∀ a : Multiset ℕ, P a → answer.prod ≥ a.prod)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q55
find_all (answer : Multiset ℕ) iff
  (P : Multiset ℕ → Prop)
  (hP : ∀ a, P a ↔ Multiset.card a > 0 ∧ (∀ i ∈ a, i > 0) ∧ a.sum = 1979)
: (P answer ∧ ∀ a : Multiset ℕ, P a → answer.prod ≥ a.prod)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
