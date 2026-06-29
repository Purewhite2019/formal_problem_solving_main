import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The four-digit number $\underline{374n}$ is divisible by 18. Find the units digit $n$.

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q197.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q197
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : n ≤ 9)
  (h₁ : 18 ∣ 374 * 10 + n)
: (answer = n)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q197
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : n ≤ 9)
  (h₁ : 18 ∣ 374 * 10 + n)
: (answer = n)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
