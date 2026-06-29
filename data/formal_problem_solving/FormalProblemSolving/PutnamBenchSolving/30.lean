import FormalProblemSolving.Basic
open Set Nat

/-
## Problem
Evaluate in closed form: $\frac{1}{2^{n-1}} \sum_{k < n/2} (n-2k)*{n \choose k}$.

## Answer
(fun n ↦ (1 : ℚ) / ((2 : ℚ) ^ ((n :ℕ) - 1)) * (n * (n - 1).choose ⌊n / 2⌋₊))
-/

-- # Formal Answer
noncomputable abbrev Q30.gt := fun (n : ℕ) (answer : ℕ → ℚ) => answer = fun (n : ℕ) =>
  (((1 : ℚ) / ((2 : ℚ) ^ (n - (1 : ℕ) : ℕ) : ℚ) : ℚ) *
      ((↑n : ℚ) * (↑((n - (1 : ℕ) : ℕ).choose (⌊(n / (2 : ℕ) : ℕ)⌋₊ : ℕ) : ℕ) : ℚ) : ℚ) :
    ℚ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q30
  (n : ℕ)
find (answer : ℕ → ℚ) s.t.
  (hn : 0 < n)
: ((1 : ℚ) / (2 ^ (n - 1)) * ∑ k in Finset.Icc 0 ⌊n / 2⌋₊, (n - 2 * k) * (n.choose k) = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q30
  (n : ℕ)
find_all (answer : ℕ → ℚ) iff
  (hn : 0 < n)
: ((1 : ℚ) / (2 ^ (n - 1)) * ∑ k in Finset.Icc 0 ⌊n / 2⌋₊, (n - 2 * k) * (n.choose k) = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
