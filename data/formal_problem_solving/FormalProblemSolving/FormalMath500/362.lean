import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose the polynomial
$$f(x) = a_nx^n + a_{n-1}x^{n-1} + \cdots + a_2x^2 + a_1x + a_0$$has integer coefficients, and its roots are distinct integers.

Given that $a_n=2$ and $a_0=66$, what is the least possible value of $|a_{n-1}|$?

## Answer
14
-/

-- # Formal Answer
noncomputable abbrev Q362.gt := fun (answer : ℝ) => answer = (14 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q362
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {x : ℝ | ∃ (n : ℕ) (r : Fin n → ℤ), n > 0 ∧ Function.Injective r ∧
          2 * (-1) ^ n * (∏ i, r i) = 66 ∧ x = 2 * abs (∑ i, r i)})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q362
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {x : ℝ | ∃ (n : ℕ) (r : Fin n → ℤ), n > 0 ∧ Function.Injective r ∧
          2 * (-1) ^ n * (∏ i, r i) = 66 ∧ x = 2 * abs (∑ i, r i)})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
