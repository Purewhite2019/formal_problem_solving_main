import FormalProblemSolving.Basic
open Topology Filter Nat Function

/-
## Problem
Let $S$ be a set of $n$ distinct real numbers. Let $A_S$ be the set of numbers that occur as averages of two distinct elements of $S$. For a given $n \geq 2$, what is the smallest possible number of elements in $A_S$?

## Answer
fun n ↦ 2 * n - 3
-/

-- # Formal Answer
noncomputable abbrev Q148.gt := fun (n : ℕ) (answer : ℕ → ℤ) => answer = fun (n : ℕ) => (((2 : ℤ) * (↑n : ℤ) : ℤ) - (3 : ℤ) : ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q148
  (n : ℕ)
find (answer : ℕ → ℤ) s.t.
  (hn : n ≥ 2)
  (A : Finset ℝ → Set ℝ)
  (hA : A = fun S ↦ {x | ∃ a ∈ S, ∃ b ∈ S, a ≠ b ∧ (a + b) / 2 = x})
: (IsLeast {k : ℤ | ∃ S : Finset ℝ, S.card = n ∧ k = (A S).ncard} (answer n))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q148
  (n : ℕ)
find_all (answer : ℕ → ℤ) iff
  (hn : n ≥ 2)
  (A : Finset ℝ → Set ℝ)
  (hA : A = fun S ↦ {x | ∃ a ∈ S, ∃ b ∈ S, a ≠ b ∧ (a + b) / 2 = x})
: (IsLeast {k : ℤ | ∃ S : Finset ℝ, S.card = n ∧ k = (A S).ncard} (answer n))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
