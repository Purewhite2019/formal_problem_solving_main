import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $P(x)$ be a monic polynomial of degree 3.  Suppose that $P(x)$ has remainder $R(x)$ when it is divided by $(x - 1)(x - 4),$ and remainder $2R(x)$ when it is divided by $(x - 2)(x - 3).$  Given that $P(0) = 5,$ find $P(5).$

## Answer
15
-/

-- # Formal Answer
noncomputable abbrev Q306.gt := fun (answer : ℝ) => answer = (15 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q306
find (answer : ℝ) s.t.
  (p : Polynomial ℝ)
  (r : Polynomial ℝ)
  (h_monic : p.Monic)
  (h_deg : p.natDegree = 3)
  (h0 : Polynomial.eval 0 p = 5)
  (h1 : p %ₘ ((Polynomial.X - 1) * (Polynomial.X - 4)) = r)
  (h2 : p %ₘ ((Polynomial.X - 2) * (Polynomial.X - 3)) = 2 * r)
: (answer = Polynomial.eval 5 p)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q306
find_all (answer : ℝ) iff
  (p : Polynomial ℝ)
  (r : Polynomial ℝ)
  (h_monic : p.Monic)
  (h_deg : p.natDegree = 3)
  (h0 : Polynomial.eval 0 p = 5)
  (h1 : p %ₘ ((Polynomial.X - 1) * (Polynomial.X - 4)) = r)
  (h2 : p %ₘ ((Polynomial.X - 2) * (Polynomial.X - 3)) = 2 * r)
: (answer = Polynomial.eval 5 p)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
