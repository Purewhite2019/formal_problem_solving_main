import FormalProblemSolving.Basic
open MvPolynomial Real Nat

/-
## Problem
For each positive integer $n$, let $a(n)$ be the number of zeroes in the base $3$ representation of $n$. For which positive real numbers $x$ does the series
\[
\sum_{n=1}^\infty \frac{x^{a(n)}}{n^3}
\]
converge?

## Answer
the set of positive real numbers $x$ such that $0 < x < 25$
-/

-- # Formal Answer
noncomputable abbrev Q113.gt := fun (answer : Set ℝ) => answer = ({x : ℝ | x > (0 : ℝ) ∧ x < (25 : ℝ)} : Set ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q113
find (answer : Set ℝ) s.t.
  (a : ℕ → ℕ)
  (ha : a = fun n ↦ {i | (digits 3 n).get i = 0}.ncard)
: ({x : ℝ | x > 0 ∧ Summable (fun n ↦ x ^ (a n) / (n ^ 3))} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q113
find_all (answer : Set ℝ) iff
  (a : ℕ → ℕ)
  (ha : a = fun n ↦ {i | (digits 3 n).get i = 0}.ncard)
: ({x : ℝ | x > 0 ∧ Summable (fun n ↦ x ^ (a n) / (n ^ 3))} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
