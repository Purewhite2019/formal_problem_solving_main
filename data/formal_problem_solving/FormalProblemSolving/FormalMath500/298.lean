import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the quotient when $x^6 - 3$ is divided by $x + 1.$

## Answer
x^5 - x^4 + x^3 - x^2 + x - 1
-/

-- # Formal Answer
noncomputable abbrev Q298.gt := fun (answer : Polynomial ℤ) => answer =
  (((((((Polynomial.X : Polynomial ℤ) ^ (5 : ℕ) : Polynomial ℤ) -
                      ((Polynomial.X : Polynomial ℤ) ^ (4 : ℕ) : Polynomial ℤ) :
                    Polynomial ℤ) +
                  ((Polynomial.X : Polynomial ℤ) ^ (3 : ℕ) : Polynomial ℤ) :
                Polynomial ℤ) -
              ((Polynomial.X : Polynomial ℤ) ^ (2 : ℕ) : Polynomial ℤ) :
            Polynomial ℤ) +
          (Polynomial.X : Polynomial ℤ) :
        Polynomial ℤ) -
      (1 : Polynomial ℤ) :
    Polynomial ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q298
find (answer : Polynomial ℤ) s.t.
: (answer = (Polynomial.X ^ 6 - 3) /ₘ (Polynomial.X + 1))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q298
find_all (answer : Polynomial ℤ) iff
: (answer = (Polynomial.X ^ 6 - 3) /ₘ (Polynomial.X + 1))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
