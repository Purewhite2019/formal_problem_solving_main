import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A strictly increasing sequence of positive integers $a_1$, $a_2$, $a_3$, $\dots$ has the property that for every positive integer $k$, the subsequence $a_{2k-1}$, $a_{2k}$, $a_{2k+1}$ is geometric and the subsequence $a_{2k}$, $a_{2k+1}$, $a_{2k+2}$ is arithmetic. Suppose that $a_{13} = 2016$. Find $a_1$.

## Answer
504
-/

-- # Formal Answer
noncomputable abbrev Q364.gt := fun (answer : ℕ) => answer = (504 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q364
find (answer : ℕ) s.t.
  (a : ℕ → ℕ)
  (h_inc : ∀ n, 0 < n → a n < a (n + 1))
  (h_pos : ∀ n, 0 < n → 0 < a n)
  (h_geom : ∀ k, 0 < k → a (2 * k - 1) * a (2 * k + 1) = (a (2 * k)) ^ 2)
  (h_arith : ∀ k, 0 < k → 2 * a (2 * k + 1) = a (2 * k) + a (2 * k + 2))
  (h13 : a 13 = 2016)
: (a 1 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q364
find_all (answer : ℕ) iff
  (a : ℕ → ℕ)
  (h_inc : ∀ n, 0 < n → a n < a (n + 1))
  (h_pos : ∀ n, 0 < n → 0 < a n)
  (h_geom : ∀ k, 0 < k → a (2 * k - 1) * a (2 * k + 1) = (a (2 * k)) ^ 2)
  (h_arith : ∀ k, 0 < k → 2 * a (2 * k + 1) = a (2 * k) + a (2 * k + 2))
  (h13 : a 13 = 2016)
: (a 1 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
