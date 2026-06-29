import FormalProblemSolving.Basic
open Set Nat

/-
## Problem
Let $k$ be a positive integer. Suppose that the integers $1, 2, 3, \dots, 3k+1$ are written down in random order. What is the probability that at no time during this process, the sum of the integers that have been written up to that time is a positive integer divisible by $3$? Your answer should be in closed form, but may include factorials.

## Answer
fun k ↦ (k)! * (k + 1)! / ((3 * k + 1) * (2 * k)!)
-/

-- # Formal Answer
noncomputable abbrev Q212.gt := fun (k : ℕ) (answer : ℕ → ℚ) => answer = fun (k : ℕ) =>
  (((↑(k ! : ℕ) : ℚ) * (↑((k + (1 : ℕ) : ℕ)! : ℕ) : ℚ) : ℚ) /
      ((((3 : ℚ) * (↑k : ℚ) : ℚ) + (1 : ℚ) : ℚ) * (↑(((2 : ℕ) * k : ℕ)! : ℕ) : ℚ) : ℚ) :
    ℚ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q212
  (k : ℕ)
find (answer : ℕ → ℚ) s.t.
  (kpos : k > 0)
  (perms : Set (Fin (3 * k + 1) → ℤ))
  (goodperms : Set (Fin (3 * k + 1) → ℤ))
  (hgoodperms : goodperms = {f ∈ perms | ¬∃ j : Fin (3 * k + 1), 3 ∣ ∑ i : Fin (3 * k + 1), ite (i ≤ j) (f i) 0})
  (hperms : perms = {f : Fin (3 * k + 1) → ℤ | ∀ y ∈ Icc 1 (3 * k + 1), ∃! x : Fin (3 * k + 1), f x = y})
: (goodperms.ncard = perms.ncard * (answer k))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q212
  (k : ℕ)
find_all (answer : ℕ → ℚ) iff
  (kpos : k > 0)
  (perms : Set (Fin (3 * k + 1) → ℤ))
  (goodperms : Set (Fin (3 * k + 1) → ℤ))
  (hgoodperms : goodperms = {f ∈ perms | ¬∃ j : Fin (3 * k + 1), 3 ∣ ∑ i : Fin (3 * k + 1), ite (i ≤ j) (f i) 0})
  (hperms : perms = {f : Fin (3 * k + 1) → ℤ | ∀ y ∈ Icc 1 (3 * k + 1), ∃! x : Fin (3 * k + 1), f x = y})
: (goodperms.ncard = perms.ncard * (answer k))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
