import FormalProblemSolving.Basic
open Set Topology Filter Polynomial

/-
## Problem
Let $F$ be a finite field with $n$ elements, and assume $n$ is odd. Suppose $x^2 + bx + c$ is an irreducible polynomial over $F$. For how many elements $d \in F$ is $x^2 + bx + c + d$ irreducible?

## Answer
fun n : ℕ ↦ (n - (1 : ℤ)) / 2
-/

-- # Formal Answer
noncomputable abbrev Q60.gt := fun (n : ℕ) (answer : ℕ → ℤ) => answer = fun (n : ℕ) => (((↑n : ℤ) - (1 : ℤ) : ℤ) / (2 : ℤ) : ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q60
  (n : ℕ)
find (answer : ℕ → ℤ) s.t.
  (F : Type*)
  (_ : Field F)
  (_ : Fintype F)
  (hn : n = Fintype.card F)
  (nodd : Odd n)
  (b : F)
  (c : F)
  (p : Polynomial F)
  (hp : p = X ^ 2 + (C b) * X + (C c) ∧ Irreducible p)
: ({d : F | Irreducible (p + (C d))}.ncard = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q60
  (n : ℕ)
find_all (answer : ℕ → ℤ) iff
  (F : Type*)
  (_ : Field F)
  (_ : Fintype F)
  (hn : n = Fintype.card F)
  (nodd : Odd n)
  (b : F)
  (c : F)
  (p : Polynomial F)
  (hp : p = X ^ 2 + (C b) * X + (C c) ∧ Irreducible p)
: ({d : F | Irreducible (p + (C d))}.ncard = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
