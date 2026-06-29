import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $\lambda$ be a constant, $0 \le \lambda \le 4,$ and let $f : [0,1] \to [0,1]$ be defined by
\[f(x) = \lambda x(1 - x).\]Find the values of $\lambda,$ $0 \le \lambda \le 4,$ for which there exists an $x \in [0,1]$ such that $f(x) \neq x$ but $f(f(x)) = x.$

## Answer
(3,4]
-/

-- # Formal Answer
noncomputable abbrev Q338.gt := fun (answer : Set ℝ) => answer = ({y : ℝ | (3 : ℝ) < y ∧ y ≤ (4 : ℝ)} : Set ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q338
find (answer : Set ℝ) s.t.
: (answer = {lam : ℝ | 0 ≤ lam ∧ lam ≤ 4 ∧
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ (lam * x * (1 - x) ≠ x) ∧ (lam * (lam * x * (1 - x)) * (1 - lam * x * (1 - x)) = x))})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q338
find_all (answer : Set ℝ) iff
: (answer = {lam : ℝ | 0 ≤ lam ∧ lam ≤ 4 ∧
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ (lam * x * (1 - x) ≠ x) ∧ (lam * (lam * x * (1 - x)) * (1 - lam * x * (1 - x)) = x))})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
