import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A sequence $(a_1,b_1)$, $(a_2,b_2)$, $(a_3,b_3)$, $\ldots$ of points in the coordinate plane satisfies

$(a_{n + 1}, b_{n + 1}) = (\sqrt {3}a_n - b_n, \sqrt {3}b_n + a_n)$  for $n = 1,2,3,\ldots$.

Suppose that $(a_{100},b_{100}) = (2,4)$.  What is $a_1 + b_1$?

$\mathrm{(A)}\ -\frac{1}{2^{97}}\qquad\mathrm{(B)}\ -\frac{1}{2^{99}}\qquad\mathrm{(C)}\ 0\qquad\mathrm{(D)}\ \frac{1}{2^{98}}\qquad\mathrm{(E)}\ \frac{1}{2^{96}}$

## Answer
\frac{1}{2^{98}}
-/

-- # Formal Answer
noncomputable abbrev Q342.gt := fun (answer : ℝ) => answer = ((1 : ℝ) / ((2 : ℝ) ^ (98 : ℕ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q342
find (answer : ℝ) s.t.
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, n ≥ 1 → a (n+1) = Real.sqrt 3 * a n - b n)
  (h₁ : ∀ n : ℕ, n ≥ 1 → b (n+1) = Real.sqrt 3 * b n + a n)
  (h₂ : a 100 = 2)
  (h₃ : b 100 = 4)
: (answer = a 1 + b 1)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q342
find_all (answer : ℝ) iff
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, n ≥ 1 → a (n+1) = Real.sqrt 3 * a n - b n)
  (h₁ : ∀ n : ℕ, n ≥ 1 → b (n+1) = Real.sqrt 3 * b n + a n)
  (h₂ : a 100 = 2)
  (h₃ : b 100 = 4)
: (answer = a 1 + b 1)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
