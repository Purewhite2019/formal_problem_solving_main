import FormalProblemSolving.Basic

/-
## Problem
Find all positive integers $n < 10^{100}$ for which simultaneously $n$ divides $2^n$, $n-1$ divides $2^n-1$, and $n-2$ divides $2^n - 2$.

## Answer
{2^2, 2^4, 2^16, 2^256}
-/

-- # Formal Answer
noncomputable abbrev Q280.gt := fun (answer : Set ℕ) => answer =
  (Insert.insert ((2 : ℕ) ^ (2 : ℕ) : ℕ)
      (Insert.insert ((2 : ℕ) ^ (4 : ℕ) : ℕ)
          (Insert.insert ((2 : ℕ) ^ (16 : ℕ) : ℕ) ({((2 : ℕ) ^ (256 : ℕ) : ℕ)} : Set ℕ) : Set ℕ) :
        Set ℕ) :
    Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q280
find (answer : Set ℕ) s.t.
  (n : ℕ)
  (hn : 0 < n)
: ((n < 10^100 ∧ ((n : ℤ) ∣ (2^n : ℤ) ∧ (n - 1 : ℤ) ∣ (2^n - 1 : ℤ) ∧ (n - 2 : ℤ) ∣ (2^n - 2 : ℤ))) ↔ n ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q280
find_all (answer : Set ℕ) iff
  (n : ℕ)
  (hn : 0 < n)
: ((n < 10^100 ∧ ((n : ℤ) ∣ (2^n : ℤ) ∧ (n - 1 : ℤ) ∣ (2^n - 1 : ℤ) ∧ (n - 2 : ℤ) ∣ (2^n - 2 : ℤ))) ↔ n ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
