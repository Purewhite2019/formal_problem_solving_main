import FormalProblemSolving.Basic
open Topology Filter Nat

/-
## Problem
For a given positive integer $m$, find all triples $(n, x, y)$ of positive integers, with $n$ relatively prime to $m$, which satisfy
\[
(x^2 + y^2)^m = (xy)^n.
\]

## Answer
$\emptyset$ if $m$ is odd, otherwise $\{(m + 1, 2^{m/2}, 2^{m/2})\}$
-/

-- # Formal Answer
noncomputable abbrev Q146.gt := fun (m : ℕ) (answer : ℕ → Set (ℕ × ℕ × ℕ)) => answer = fun (m : ℕ) =>
  (if Odd m then (∅ : Set (ℕ × ℕ × ℕ))
    else
      ({(((m + (1 : ℕ) : ℕ), ((((2 : ℕ) ^ (m / (2 : ℕ) : ℕ) : ℕ), ((2 : ℕ) ^ (m / (2 : ℕ) : ℕ) : ℕ)) : ℕ × ℕ)) :
            ℕ × ℕ × ℕ)} :
        Set (ℕ × ℕ × ℕ)) :
    Set (ℕ × ℕ × ℕ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q146
  (m : ℕ)
find (answer : ℕ → Set (ℕ × ℕ × ℕ)) s.t.
  (mpos : m > 0)
  (S : Set (ℕ × ℕ × ℕ))
  (hS : ∀ n x y : ℕ, (n, x, y) ∈ S ↔ n > 0 ∧ x > 0 ∧ y > 0 ∧ Coprime n m ∧ (x ^ 2 + y ^ 2) ^ m = (x * y) ^ n)
: (S = answer m)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q146
  (m : ℕ)
find_all (answer : ℕ → Set (ℕ × ℕ × ℕ)) iff
  (mpos : m > 0)
  (S : Set (ℕ × ℕ × ℕ))
  (hS : ∀ n x y : ℕ, (n, x, y) ∈ S ↔ n > 0 ∧ x > 0 ∧ y > 0 ∧ Coprime n m ∧ (x ^ 2 + y ^ 2) ^ m = (x * y) ^ n)
: (S = answer m)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
