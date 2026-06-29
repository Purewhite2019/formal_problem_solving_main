import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What integer $n$ satisfies $0\le n<18$ and $$n\equiv -11213141\pmod{18}~?$$

## Answer
13
-/

-- # Formal Answer
noncomputable abbrev Q284.gt := fun (answer : ℕ) => answer = (13 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q284
find (answer : ℕ) s.t.
  (n : ℕ)
  (hn : 0 ≤ n ∧ n < 18 ∧ n ≡ 18 - (11213141 % 18) [MOD 18])
: (answer = n)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q284
find_all (answer : ℕ) iff
  (n : ℕ)
  (hn : 0 ≤ n ∧ n < 18 ∧ n ≡ 18 - (11213141 % 18) [MOD 18])
: (answer = n)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
