import FormalProblemSolving.Basic
open Filter Topology Set

/-
## Problem
For a positive integer $n$, define $d(n)$ to be the sum of the digits of $n$ when written in binary (for example, $d(13) = 1+1+0+1=3)$. Let
\[
S = \sum_{k=1}^{2020} (-1)^{d(k)} k^3.
\]
Determine $S$ modulo 2020.

## Answer
1990
-/

-- # Formal Answer
noncomputable abbrev Q295.gt := fun (answer : ℕ) => answer = (1990 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q295
find (answer : ℕ) s.t.
  (d : ℕ → ℕ)
  (S : ℤ)
  (hd : d = fun n : ℕ => ∑ i : Fin (Nat.digits 2 n).length, (Nat.digits 2 n)[i]!)
  (hS : S = ∑ k : Icc 1 2020, ((-1 : ℤ)^(d k))*(k : ℤ)^3)
: (S % 2020 = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q295
find_all (answer : ℕ) iff
  (d : ℕ → ℕ)
  (S : ℤ)
  (hd : d = fun n : ℕ => ∑ i : Fin (Nat.digits 2 n).length, (Nat.digits 2 n)[i]!)
  (hS : S = ∑ k : Icc 1 2020, ((-1 : ℤ)^(d k))*(k : ℤ)^3)
: (S % 2020 = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
