import FormalProblemSolving.Basic
open Nat

/-
## Problem
Let $S$ be a non-empty set with an associative operation that is left and right cancellative ($xy=xz$ implies $y=z$, and $yx=zx$ implies $y=z$). Assume that for every $a$ in $S$ the set $\{a^n:\,n=1, 2, 3, \ldots\}$ is finite. Must $S$ be a group?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q127.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q127
find (answer : Prop) s.t.
  (pow : (S : Type) → ℕ → S → S)
  (hpow1 : ∀ (S : Type) (_ : Semigroup S), ∀ s : S, pow S 1 s = s)
  (hpown : ∀ (S : Type) (_ : Semigroup S), ∀ s : S, ∀ n > 0, pow S (n + 1) s = s * (pow S n s))
: (((∀ (S : Type) (_ : Nonempty S) (_ : Semigroup S) (_ : IsCancelMul S), (∀ a : S, Finite {x | ∃ n : ℕ, n > 0 ∧ pow S n a = x}) → ∃ (_ : Group S), True) ↔ answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q127
find_all (answer : Prop) iff
  (pow : (S : Type) → ℕ → S → S)
  (hpow1 : ∀ (S : Type) (_ : Semigroup S), ∀ s : S, pow S 1 s = s)
  (hpown : ∀ (S : Type) (_ : Semigroup S), ∀ s : S, ∀ n > 0, pow S (n + 1) s = s * (pow S n s))
: (((∀ (S : Type) (_ : Nonempty S) (_ : Semigroup S) (_ : IsCancelMul S), (∀ a : S, Finite {x | ∃ n : ℕ, n > 0 ∧ pow S n a = x}) → ∃ (_ : Group S), True) ↔ answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
