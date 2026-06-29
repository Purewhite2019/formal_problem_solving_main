import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For how many positive integers $n>1$ is it true that $2^{24}$ is a perfect $n^{\text{th}}$ power?

## Answer
7
-/

-- # Formal Answer
noncomputable abbrev Q247.gt := fun (answer : ℕ) => answer = (7 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q247
find (answer : ℕ) s.t.
  (n : ℕ)
  (S : Set ℕ)
  (hS : S = {n : ℕ | n > 1 ∧ ∃ (k : ℕ), k^n = 2^24})
: (answer = S.encard)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q247
find_all (answer : ℕ) iff
  (n : ℕ)
  (S : Set ℕ)
  (hS : S = {n : ℕ | n > 1 ∧ ∃ (k : ℕ), k^n = 2^24})
: (answer = S.encard)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
