import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $n = 11$, then what is $\left(\frac{1}{4}\right)^{n+1} \cdot 2^{2n}$?

## Answer
\frac{1}{4}
-/

-- # Formal Answer
noncomputable abbrev Q359.gt := fun (answer : ℚ) => answer = (1 / 4 : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q359
find (answer : ℚ) s.t.
  (n : ℕ)
  (h₀ : n = 11)
: (answer = (1 / 4)^(n + 1) * 2^(2 * n))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q359
find_all (answer : ℚ) iff
  (n : ℕ)
  (h₀ : n = 11)
: (answer = (1 / 4)^(n + 1) * 2^(2 * n))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
