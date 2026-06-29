import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The polynomial $x^3 - 3x^2 + 4x - 1$ is a factor of $x^9 + px^6 + qx^3 + r.$  Enter the ordered triple $(p,q,r).$

## Answer
(6,31,-1)
-/

-- # Formal Answer
noncomputable abbrev Q317.gt := fun (answer : ℤ × ℤ × ℤ) => answer = (((6 : ℤ), (((31 : ℤ), (-1 : ℤ)) : ℤ × ℤ)) : ℤ × ℤ × ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q317
find (answer : ℤ × ℤ × ℤ) s.t.
: (∃ (Q : Polynomial ℤ),
    Polynomial.X ^ 9 + Polynomial.C (answer.1) * Polynomial.X ^ 6 +
      Polynomial.C (answer.2.1) * Polynomial.X ^ 3 + Polynomial.C (answer.2.2) =
      (Polynomial.X ^ 3 - 3 * Polynomial.X ^ 2 + 4 * Polynomial.X - 1) * Q)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q317
find_all (answer : ℤ × ℤ × ℤ) iff
: (∃ (Q : Polynomial ℤ),
    Polynomial.X ^ 9 + Polynomial.C (answer.1) * Polynomial.X ^ 6 +
      Polynomial.C (answer.2.1) * Polynomial.X ^ 3 + Polynomial.C (answer.2.2) =
      (Polynomial.X ^ 3 - 3 * Polynomial.X ^ 2 + 4 * Polynomial.X - 1) * Q)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
