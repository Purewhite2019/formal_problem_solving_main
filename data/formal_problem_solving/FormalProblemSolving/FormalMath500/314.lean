import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x) = x^{10}+5x^9-8x^8+7x^7-x^6-12x^5+4x^4-8x^3+12x^2-5x-5$.

Without using long division (which would be horribly nasty!), find the remainder when $f(x)$ is divided by $x^2-1$.

## Answer
-13x+3
-/

-- # Formal Answer
noncomputable abbrev Q314.gt := fun (answer : Polynomial ℝ) => answer = (((-13 : Polynomial ℝ) * (Polynomial.X : Polynomial ℝ) : Polynomial ℝ) + (3 : Polynomial ℝ) : Polynomial ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q314
find (answer : Polynomial ℝ) s.t.
: (answer = (Polynomial.X ^ 10 + 5 * Polynomial.X ^ 9 - 8 * Polynomial.X ^ 8 + 7 * Polynomial.X ^ 7
    - Polynomial.X ^ 6 - 12 * Polynomial.X ^ 5 + 4 * Polynomial.X ^ 4 - 8 * Polynomial.X ^ 3 + 12 * Polynomial.X ^ 2
    - 5 * Polynomial.X - 5) %ₘ (Polynomial.X ^ 2 - 1))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q314
find_all (answer : Polynomial ℝ) iff
: (answer = (Polynomial.X ^ 10 + 5 * Polynomial.X ^ 9 - 8 * Polynomial.X ^ 8 + 7 * Polynomial.X ^ 7
    - Polynomial.X ^ 6 - 12 * Polynomial.X ^ 5 + 4 * Polynomial.X ^ 4 - 8 * Polynomial.X ^ 3 + 12 * Polynomial.X ^ 2
    - 5 * Polynomial.X - 5) %ₘ (Polynomial.X ^ 2 - 1))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
