import FormalProblemSolving.Basic
open Topology Filter Polynomial

/-
## Problem
For what integer $a$ does $x^2-x+a$ divide $x^{13}+x+90$?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q4.gt := fun (answer : ℤ) => answer = (2 : ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q4
find (answer : ℤ) s.t.
  (a : ℤ)
  (h_div : Polynomial.X^2 - Polynomial.X + (Polynomial.C a) ∣ (Polynomial.X ^ 13 + Polynomial.X + (Polynomial.C 90)))
: (answer = a)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q4
find_all (answer : ℤ) iff
  (a : ℤ)
  (h_div : Polynomial.X^2 - Polynomial.X + (Polynomial.C a) ∣ (Polynomial.X ^ 13 + Polynomial.X + (Polynomial.C 90)))
: (answer = a)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
