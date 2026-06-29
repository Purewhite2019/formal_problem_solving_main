import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $n$ be a positive integer.  Simplify the expression
\[\frac{(2^4 + \frac{1}{4})(4^4 + \frac{1}{4}) \dotsm [(2n)^4 + \frac{1}{4}]}{(1^4 + \frac{1}{4})(3^4 + \frac{1}{4}) \dotsm [(2n - 1)^4 + \frac{1}{4}]}.\]

## Answer
8n^2 + 4n + 1
-/

-- # Formal Answer
noncomputable abbrev Q358.gt := fun (n : ℕ) (answer : ℕ) => answer = ((((8 : ℕ) * (n ^ (2 : ℕ) : ℕ) : ℕ) + ((4 : ℕ) * n : ℕ) : ℕ) + (1 : ℕ) : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q358
  (n : ℕ)
find (answer : ℕ) s.t.
  (h₀ : n > 0)
  (numerator : ℚ)
  (denominator : ℚ)
  (expression : ℕ)
  (h_numerator : numerator = ∏ k in Finset.range n, ((2 * k + 2)^4 + (1:ℚ)/4))
  (h_denominator : denominator = ∏ k in Finset.range n, ((2 * k + 1)^4 + (1:ℚ)/4))
  (h_expression : expression = numerator / denominator)
: (expression = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q358
  (n : ℕ)
find_all (answer : ℕ) iff
  (h₀ : n > 0)
  (numerator : ℚ)
  (denominator : ℚ)
  (expression : ℕ)
  (h_numerator : numerator = ∏ k in Finset.range n, ((2 * k + 2)^4 + (1:ℚ)/4))
  (h_denominator : denominator = ∏ k in Finset.range n, ((2 * k + 1)^4 + (1:ℚ)/4))
  (h_expression : expression = numerator / denominator)
: (expression = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
