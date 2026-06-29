import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The operation @ is defined as $\frac{m}{n}@\frac{p}{q} = (m)(p)\left(\frac{q}{n}\right)$ for simplified fractions $\frac{p}{q}$.  What is the simplified value of $\frac{7}{30}@\frac{10}{21}$?

## Answer
49
-/

-- # Formal Answer
noncomputable abbrev Q178.gt := fun (answer : ℚ) => answer = (49 : ℚ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q178
find (answer : ℚ) s.t.
  (at_op : ℤ → ℤ → ℤ → ℤ → ℚ)
  (h_def : ∀ m n p q, at_op m n p q = (m * p) * ((q : ℚ) / n))
  (h_simp : Nat.gcd 10 21 = 1)
: (at_op 7 30 10 21 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q178
find_all (answer : ℚ) iff
  (at_op : ℤ → ℤ → ℤ → ℤ → ℚ)
  (h_def : ∀ m n p q, at_op m n p q = (m * p) * ((q : ℚ) / n))
  (h_simp : Nat.gcd 10 21 = 1)
: (at_op 7 30 10 21 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
