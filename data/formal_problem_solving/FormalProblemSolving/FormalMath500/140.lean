import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Consider the geometric series $4+\frac{12}{a}+\frac{36}{a^2}+\cdots$. If the sum is a perfect square, what is the smallest possible value of $a$ where $a$ is a positive integer?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q140.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q140
find (answer : ℕ) s.t.
: (0 < answer) ∧ (3 < answer) ∧ (∃ (s : ℕ), (s : ℚ)^2 = 4 * (answer : ℚ) / ((answer : ℚ) - 3)) ∧ (∀ (a : ℕ), 0 < a → 3 < a → (∃ (s : ℕ), (s : ℚ)^2 = 4 * (a : ℚ) / ((a : ℚ) - 3)) → answer ≤ a)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q140
find_all (answer : ℕ) iff
: (0 < answer) ∧ (3 < answer) ∧ (∃ (s : ℕ), (s : ℚ)^2 = 4 * (answer : ℚ) / ((answer : ℚ) - 3)) ∧ (∀ (a : ℕ), 0 < a → 3 < a → (∃ (s : ℕ), (s : ℚ)^2 = 4 * (a : ℚ) / ((a : ℚ) - 3)) → answer ≤ a)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
