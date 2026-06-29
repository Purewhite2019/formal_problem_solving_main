import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the probability that a randomly selected integer in the set $$\{1,2,3,\ldots,100\}$$  is divisible by 2 and not divisible by 3? Express your answer as a common fraction.

## Answer
\dfrac{17}{50}
-/

-- # Formal Answer
noncomputable abbrev Q34.gt := fun (answer : ℝ) => answer = (17 / 50 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q34
find (answer : ℝ) s.t.
: (answer = (Nat.card {n : ℕ | 1 ≤ n ∧ n ≤ 100 ∧ n % 2 = 0 ∧ n % 3 ≠ 0} : ℝ) / (Nat.card {n : ℕ | 1 ≤ n ∧ n ≤ 100} : ℝ))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q34
find_all (answer : ℝ) iff
: (answer = (Nat.card {n : ℕ | 1 ≤ n ∧ n ≤ 100 ∧ n % 2 = 0 ∧ n % 3 ≠ 0} : ℝ) / (Nat.card {n : ℕ | 1 ≤ n ∧ n ≤ 100} : ℝ))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
