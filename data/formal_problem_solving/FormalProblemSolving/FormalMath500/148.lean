import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose that $f$ is a polynomial such that \[(x-1)\cdot f(x)=3x^4+x^3 - 25x^2 +38x -17.\]What is the degree of $f$?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q148.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q148
find (answer : ℕ) s.t.
  (f : Polynomial ℝ)
  (hf : (Polynomial.X - 1) * f = 3 * Polynomial.X ^ 4 + Polynomial.X ^ 3 - 25 * Polynomial.X ^ 2 + 38 * Polynomial.X - 17)
: (Polynomial.degree f = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q148
find_all (answer : ℕ) iff
  (f : Polynomial ℝ)
  (hf : (Polynomial.X - 1) * f = 3 * Polynomial.X ^ 4 + Polynomial.X ^ 3 - 25 * Polynomial.X ^ 2 + 38 * Polynomial.X - 17)
: (Polynomial.degree f = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
