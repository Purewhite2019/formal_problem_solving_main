import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For how many real values of $x$ is $\sqrt{120-\sqrt{x}}$ an integer?

## Answer
11
-/

-- # Formal Answer
noncomputable abbrev Q159.gt := fun (answer : ℕ) => answer = (11 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q159
find (answer : ℕ) s.t.
: (Nat.card {x : ℝ | 0 ≤ x ∧ 0 ≤ 120 - Real.sqrt x ∧ ∃ (n : ℤ), Real.sqrt (120 - Real.sqrt x) = n} = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q159
find_all (answer : ℕ) iff
: (Nat.card {x : ℝ | 0 ≤ x ∧ 0 ≤ 120 - Real.sqrt x ∧ ∃ (n : ℤ), Real.sqrt (120 - Real.sqrt x) = n} = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
