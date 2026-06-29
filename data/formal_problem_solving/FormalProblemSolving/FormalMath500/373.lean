import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If
\[\sqrt[3]{2} = a + \cfrac{1}{b + \cfrac{1}{c + \cfrac{1}{d + \dotsb}}},\]where $a,$ $b,$ $c,$ $d$ are positive integers, compute $b.$

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q373.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q373
find (answer : ℕ) s.t.
  (x : ℝ)
  (hx : x = Real.rpow 2 (1/3 : ℝ))
  (a : ℤ)
  (ha : a = Int.floor x)
  (b : ℤ)
  (hb : b = Int.floor (1 / (x - a)))
: (answer = b.toNat)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q373
find_all (answer : ℕ) iff
  (x : ℝ)
  (hx : x = Real.rpow 2 (1/3 : ℝ))
  (a : ℤ)
  (ha : a = Int.floor x)
  (b : ℤ)
  (hb : b = Int.floor (1 / (x - a)))
: (answer = b.toNat)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
