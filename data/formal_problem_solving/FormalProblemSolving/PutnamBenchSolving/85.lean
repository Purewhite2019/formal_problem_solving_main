import FormalProblemSolving.Basic

/-
## Problem
Express $\sum_{k=1}^\infty (6^k/(3^{k+1}-2^{k+1})(3^k-2^k))$ as a rational number.

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q85.gt := fun (answer : ℚ) => answer = (2 : ℚ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q85
find (answer : ℚ) s.t.
: (∑' k : Set.Ici 1, (6 ^ (k : ℕ) / ((3 ^ ((k : ℕ) + 1) - 2 ^ ((k : ℕ) + 1)) * (3 ^ (k : ℕ) - 2 ^ (k : ℕ)))) = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q85
find_all (answer : ℚ) iff
: (∑' k : Set.Ici 1, (6 ^ (k : ℕ) / ((3 ^ ((k : ℕ) + 1) - 2 ^ ((k : ℕ) + 1)) * (3 ^ (k : ℕ) - 2 ^ (k : ℕ)))) = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
