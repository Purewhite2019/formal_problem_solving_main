import FormalProblemSolving.Basic
open Set Nat Polynomial Filter Topology

/-
## Problem
For a set with $1000$ elements, how many subsets are there whose candinality is respectively $\equiv 0 \bmod 3, \equiv 1 \bmod 3, \equiv 2 \bmod 3$?

## Answer
((2^1000 - 1)/3, (2^1000 - 1)/3, 1 + (2^1000 - 1)/3)
-/

-- # Formal Answer
noncomputable abbrev Q32.gt := fun (answer : ℕ × ℕ × ℕ) => answer =
  ((((((2 : ℕ) ^ (1000 : ℕ) : ℕ) - (1 : ℕ) : ℕ) / (3 : ℕ) : ℕ),
      ((((((2 : ℕ) ^ (1000 : ℕ) : ℕ) - (1 : ℕ) : ℕ) / (3 : ℕ) : ℕ),
          ((1 : ℕ) + ((((2 : ℕ) ^ (1000 : ℕ) : ℕ) - (1 : ℕ) : ℕ) / (3 : ℕ) : ℕ) : ℕ)) :
        ℕ × ℕ)) :
    ℕ × ℕ × ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q32
find (answer : ℕ × ℕ × ℕ) s.t.
  (n : ℤ)
  (hn : n = 1000)
  (count0 : ℕ)
  (count1 : ℕ)
  (count2 : ℕ)
  (hcount0 : count0 = {S | S ⊆ Finset.Icc 1 n ∧ S.card ≡ 0 [MOD 3]}.ncard)
  (hcount1 : count1 = {S | S ⊆ Finset.Icc 1 n ∧ S.card ≡ 1 [MOD 3]}.ncard)
  (hcount2 : count2 = {S | S ⊆ Finset.Icc 1 n ∧ S.card ≡ 2 [MOD 3]}.ncard)
: ((count0, count1, count2) = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q32
find_all (answer : ℕ × ℕ × ℕ) iff
  (n : ℤ)
  (hn : n = 1000)
  (count0 : ℕ)
  (count1 : ℕ)
  (count2 : ℕ)
  (hcount0 : count0 = {S | S ⊆ Finset.Icc 1 n ∧ S.card ≡ 0 [MOD 3]}.ncard)
  (hcount1 : count1 = {S | S ⊆ Finset.Icc 1 n ∧ S.card ≡ 1 [MOD 3]}.ncard)
  (hcount2 : count2 = {S | S ⊆ Finset.Icc 1 n ∧ S.card ≡ 2 [MOD 3]}.ncard)
: ((count0, count1, count2) = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
