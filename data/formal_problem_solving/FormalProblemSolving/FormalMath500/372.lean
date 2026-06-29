import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the quotient when $x^9 + 1$ is divided by $x - 1.$

## Answer
x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1
-/

-- # Formal Answer
noncomputable abbrev Q372.gt := fun (answer : Polynomial ℝ) => answer =
  ((((((((((Polynomial.X : Polynomial ℝ) ^ (8 : ℕ) : Polynomial ℝ) +
                                  ((Polynomial.X : Polynomial ℝ) ^ (7 : ℕ) : Polynomial ℝ) :
                                Polynomial ℝ) +
                              ((Polynomial.X : Polynomial ℝ) ^ (6 : ℕ) : Polynomial ℝ) :
                            Polynomial ℝ) +
                          ((Polynomial.X : Polynomial ℝ) ^ (5 : ℕ) : Polynomial ℝ) :
                        Polynomial ℝ) +
                      ((Polynomial.X : Polynomial ℝ) ^ (4 : ℕ) : Polynomial ℝ) :
                    Polynomial ℝ) +
                  ((Polynomial.X : Polynomial ℝ) ^ (3 : ℕ) : Polynomial ℝ) :
                Polynomial ℝ) +
              ((Polynomial.X : Polynomial ℝ) ^ (2 : ℕ) : Polynomial ℝ) :
            Polynomial ℝ) +
          (Polynomial.X : Polynomial ℝ) :
        Polynomial ℝ) +
      (1 : Polynomial ℝ) :
    Polynomial ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q372
find (answer : Polynomial ℝ) s.t.
: (answer = (Polynomial.X ^ 9 + 1) / (Polynomial.X - 1))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q372
find_all (answer : Polynomial ℝ) iff
: (answer = (Polynomial.X ^ 9 + 1) / (Polynomial.X - 1))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
