import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What integer $n$ satisfies $0\le n<{101}$ and $$123456\equiv n\pmod {101}~?$$

## Answer
34
-/

-- # Formal Answer
noncomputable abbrev Q309.gt := fun (answer : ℤ) => answer = (34 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q309
find (answer : ℤ) s.t.
: (0 ≤ answer) ∧ (answer < 101) ∧ ((123456 : ℤ) ≡ answer [ZMOD 101])
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q309
find_all (answer : ℤ) iff
: (0 ≤ answer) ∧ (answer < 101) ∧ ((123456 : ℤ) ≡ answer [ZMOD 101])
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
