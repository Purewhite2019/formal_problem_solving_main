import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The solutions of the equation $z^4+4z^3i-6z^2-4zi-i=0$ are the vertices of a convex polygon in the complex plane.  The area of this polygon can be expressed in the form $p^{a/b},$ where $a,$ $b,$ $p$ are positive integers, $p$ is prime, and $a$ and $b$ are relatively prime.  Find $a + b + p.$

## Answer
11
-/

-- # Formal Answer
noncomputable abbrev Q86.gt := fun (answer : ℕ) => answer = (11 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q86
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (p : ℕ)
  (z1 : ℂ)
  (z2 : ℂ)
  (z3 : ℂ)
  (z4 : ℂ)
  (h_factor : ∀ z : ℂ, (z - z1) * (z - z2) * (z - z3) * (z - z4) = z^4 + 4 * Complex.I * z^3 - 6 * z^2 - 4 * Complex.I * z - Complex.I)
  (h_distinct : z1 ≠ z2 ∧ z1 ≠ z3 ∧ z1 ≠ z4 ∧ z2 ≠ z3 ∧ z2 ≠ z4 ∧ z3 ≠ z4)
  (h_pos1 : 0 < (z2.re - z1.re) * (z3.im - z1.im) - (z3.re - z1.re) * (z2.im - z1.im))
  (h_pos2 : 0 < (z3.re - z2.re) * (z4.im - z2.im) - (z4.re - z2.re) * (z3.im - z2.im))
  (h_pos3 : 0 < (z4.re - z3.re) * (z1.im - z3.im) - (z1.re - z3.re) * (z4.im - z3.im))
  (h_pos4 : 0 < (z1.re - z4.re) * (z2.im - z4.im) - (z2.re - z4.re) * (z1.im - z4.im))
  (area : ℝ)
  (harea : area = ((z2.re - z1.re) * (z3.im - z1.im) - (z3.re - z1.re) * (z2.im - z1.im) +
                   (z3.re - z1.re) * (z4.im - z1.im) - (z4.re - z1.re) * (z3.im - z1.im)) / 2)
  (ha : a > 0)
  (hb : b > 0)
  (hp : Nat.Prime p)
  (hgcd : Nat.gcd a b = 1)
  (habp : area = (p : ℝ) ^ ((a : ℝ) / (b : ℝ)))
: (answer = a + b + p)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q86
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (p : ℕ)
  (z1 : ℂ)
  (z2 : ℂ)
  (z3 : ℂ)
  (z4 : ℂ)
  (h_factor : ∀ z : ℂ, (z - z1) * (z - z2) * (z - z3) * (z - z4) = z^4 + 4 * Complex.I * z^3 - 6 * z^2 - 4 * Complex.I * z - Complex.I)
  (h_distinct : z1 ≠ z2 ∧ z1 ≠ z3 ∧ z1 ≠ z4 ∧ z2 ≠ z3 ∧ z2 ≠ z4 ∧ z3 ≠ z4)
  (h_pos1 : 0 < (z2.re - z1.re) * (z3.im - z1.im) - (z3.re - z1.re) * (z2.im - z1.im))
  (h_pos2 : 0 < (z3.re - z2.re) * (z4.im - z2.im) - (z4.re - z2.re) * (z3.im - z2.im))
  (h_pos3 : 0 < (z4.re - z3.re) * (z1.im - z3.im) - (z1.re - z3.re) * (z4.im - z3.im))
  (h_pos4 : 0 < (z1.re - z4.re) * (z2.im - z4.im) - (z2.re - z4.re) * (z1.im - z4.im))
  (area : ℝ)
  (harea : area = ((z2.re - z1.re) * (z3.im - z1.im) - (z3.re - z1.re) * (z2.im - z1.im) +
                   (z3.re - z1.re) * (z4.im - z1.im) - (z4.re - z1.re) * (z3.im - z1.im)) / 2)
  (ha : a > 0)
  (hb : b > 0)
  (hp : Nat.Prime p)
  (hgcd : Nat.gcd a b = 1)
  (habp : area = (p : ℝ) ^ ((a : ℝ) / (b : ℝ)))
: (answer = a + b + p)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
