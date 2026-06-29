import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The function $f$ satisfies the functional equation
\[f(x) + f(y) = f(x + y) - xy - 1\]for all real numbers $x$ and $y.$  If $f(1) = 1,$ then find all integers $n$ such that $f(n) = n.$  Enter all such integers, separated by commas.

## Answer
1,-2
-/

-- # Formal Answer
noncomputable abbrev Q335.gt := fun (answer : Set ℤ) => answer = (Insert.insert (1 : ℤ) ({(-2 : ℤ)} : Set ℤ) : Set ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q335
find (answer : Set ℤ) s.t.
  (f : ℝ → ℝ)
  (h_func_eq : ∀ x y : ℝ, f x + f y = f (x + y) - x * y - 1)
  (h_f1 : f 1 = 1)
: (answer = { n : ℤ | f (n : ℝ) = (n : ℝ) })
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q335
find_all (answer : Set ℤ) iff
  (f : ℝ → ℝ)
  (h_func_eq : ∀ x y : ℝ, f x + f y = f (x + y) - x * y - 1)
  (h_f1 : f 1 = 1)
: (answer = { n : ℤ | f (n : ℝ) = (n : ℝ) })
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
