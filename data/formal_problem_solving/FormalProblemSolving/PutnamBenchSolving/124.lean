import FormalProblemSolving.Basic

/-
## Problem
How many primes among the positive integers, written as usual in base $10$, are alternating $1$'s and $0$'s, beginning and ending with $1$?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q124.gt := fun (answer : ℕ∞) => answer = (1 : ℕ∞)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q124
find (answer : ℕ∞) s.t.
  (pdigalt : List ℕ → Prop)
  (hpdigalt : ∀ l, pdigalt l ↔ Odd l.length ∧ (∀ i, l.get i = if Even (i : ℕ) then 1 else 0))
: ({p : ℕ | p.Prime ∧ pdigalt (Nat.digits 10 p)}.encard = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q124
find_all (answer : ℕ∞) iff
  (pdigalt : List ℕ → Prop)
  (hpdigalt : ∀ l, pdigalt l ↔ Odd l.length ∧ (∀ i, l.get i = if Even (i : ℕ) then 1 else 0))
: ({p : ℕ | p.Prime ∧ pdigalt (Nat.digits 10 p)}.encard = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
