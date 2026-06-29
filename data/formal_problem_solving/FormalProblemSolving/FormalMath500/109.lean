import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the equation of the plane containing the points $(0,-1,-1),$ $(-4,4,4),$ and $(4,5,1).$  Enter your answer in the form
\[Ax + By + Cz + D = 0,\]where $A,$ $B,$ $C,$ $D$ are integers such that $A > 0$ and $\gcd(|A|,|B|,|C|,|D|) = 1.$

## Answer
5x - 7y + 11z + 4 = 0
-/

-- # Formal Answer
noncomputable abbrev Q109.gt := fun (answer : ℤ × ℤ × ℤ × ℤ) => answer = (((5 : ℤ), (((-7 : ℤ), (((11 : ℤ), (4 : ℤ)) : ℤ × ℤ)) : ℤ × ℤ × ℤ)) : ℤ × ℤ × ℤ × ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q109
find (answer : ℤ × ℤ × ℤ × ℤ) s.t.
  (A : ℤ)
  (B : ℤ)
  (C : ℤ)
  (D : ℤ)
  (h₀ : A > 0)
  (h₁ : Int.gcd (Int.gcd (Int.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1)
  (h₂ : - B - C  + D = 0 ∧ -4 * A + 4 * B + 4*C + D = 0 ∧ 4 * A + 5 * B + C + D = 0)
: ((A, B, C, D) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q109
find_all (answer : ℤ × ℤ × ℤ × ℤ) iff
  (A : ℤ)
  (B : ℤ)
  (C : ℤ)
  (D : ℤ)
  (h₀ : A > 0)
  (h₁ : Int.gcd (Int.gcd (Int.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1)
  (h₂ : - B - C  + D = 0 ∧ -4 * A + 4 * B + 4*C + D = 0 ∧ 4 * A + 5 * B + C + D = 0)
: ((A, B, C, D) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
