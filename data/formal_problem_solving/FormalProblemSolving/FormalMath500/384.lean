import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $a,$ $b,$ and $c$ be distinct real numbers.  Find the degree of the polynomial
\[p(x) = \frac{(x - b)(x - c)}{(a - b)(a - c)} + \frac{(x - a)(x - c)}{(b - a)(b - c)} + \frac{(x - a)(x - b)}{(c - a)(c - b)}.\]

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q384.gt := fun (answer : ℕ) => answer = (0 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q384
find (answer : ℕ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h1 : a ≠ b)
  (h2 : a ≠ c)
  (h3 : b ≠ c)
: (answer = Polynomial.natDegree
    (Polynomial.C (1 / ((a - b) * (a - c))) * (Polynomial.X - Polynomial.C b) * (Polynomial.X - Polynomial.C c) +
     Polynomial.C (1 / ((b - a) * (b - c))) * (Polynomial.X - Polynomial.C a) * (Polynomial.X - Polynomial.C c) +
     Polynomial.C (1 / ((c - a) * (c - b))) * (Polynomial.X - Polynomial.C a) * (Polynomial.X - Polynomial.C b)))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q384
find_all (answer : ℕ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h1 : a ≠ b)
  (h2 : a ≠ c)
  (h3 : b ≠ c)
: (answer = Polynomial.natDegree
    (Polynomial.C (1 / ((a - b) * (a - c))) * (Polynomial.X - Polynomial.C b) * (Polynomial.X - Polynomial.C c) +
     Polynomial.C (1 / ((b - a) * (b - c))) * (Polynomial.X - Polynomial.C a) * (Polynomial.X - Polynomial.C c) +
     Polynomial.C (1 / ((c - a) * (c - b))) * (Polynomial.X - Polynomial.C a) * (Polynomial.X - Polynomial.C b)))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
