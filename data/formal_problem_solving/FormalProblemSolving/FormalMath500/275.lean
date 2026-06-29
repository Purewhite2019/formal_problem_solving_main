import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the last nonzero digit to the right of the decimal point in the decimal expansion of $\frac{137}{500}$?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q275.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q275
find (answer : ℕ) s.t.
  (n : ℕ)
  (hn : n = 137)
  (d : ℕ)
  (hd : d = 500)
: (answer = ((Nat.digits 10 (n * 10 ^ 3 / d)).find? (fun d => d ≠ 0)).get (by
      rw [hn, hd]
      native_decide))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q275
find_all (answer : ℕ) iff
  (n : ℕ)
  (hn : n = 137)
  (d : ℕ)
  (hd : d = 500)
: (answer = ((Nat.digits 10 (n * 10 ^ 3 / d)).find? (fun d => d ≠ 0)).get (by
      rw [hn, hd]
      native_decide))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
