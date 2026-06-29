import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
When converting $\frac{31}{11111}$ to a decimal, the decimal turns out to be a repeating decimal. How many digits repeat in this repeating decimal?

For example, if you get the repeating decimal $0.\overline{123},$ then your answer should be $3,$ and if you get $0.436\overline{7},$ your answer should be $1.$

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q32.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q32
find (answer : ℕ) s.t.
  (n : ℕ)
  (h_repeating : n > 0 ∧ (10^n - 1) % 11111 = 0)
  (h_minimal : ∀ (m : ℕ), (m > 0 ∧ m < n) → (10^m - 1) % 11111 ≠ 0)
: (n = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q32
find_all (answer : ℕ) iff
  (n : ℕ)
  (h_repeating : n > 0 ∧ (10^n - 1) % 11111 = 0)
  (h_minimal : ∀ (m : ℕ), (m > 0 ∧ m < n) → (10^m - 1) % 11111 ≠ 0)
: (n = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
