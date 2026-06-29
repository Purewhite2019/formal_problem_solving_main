import FormalProblemSolving.Basic
open Polynomial Filter Topology Real Set Nat

/-
## Problem
Find the smallest positive integer $j$ such that for every polynomial $p(x)$ with integer coefficients and for every integer $k$, the integer \[ p^{(j)}(k) = \left. \frac{d^j}{dx^j} p(x) \right|_{x=k} \] (the $j$-th derivative of $p(x)$ at $k$) is divisible by 2016.

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q263.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q263
find (answer : ℕ) s.t.
: (IsLeast {j : ℕ | 0 < j ∧ ∀ P : ℤ[X], ∀ k : ℤ, 2016 ∣ (derivative^[j] P).eval k} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q263
find_all (answer : ℕ) iff
: (IsLeast {j : ℕ | 0 < j ∧ ∀ P : ℤ[X], ∀ k : ℤ, 2016 ∣ (derivative^[j] P).eval k} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
