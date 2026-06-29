import FormalProblemSolving.Basic
open Set Function Metric

/-
## Problem
Given a point $(a,b)$ with $0<b<a$, determine the minimum perimeter of a triangle with one vertex at $(a,b)$, one on the $x$-axis, and one on the line $y=x$.  You may assume that a triangle of minimum perimeter exists.

## Answer
$\sqrt{2a^2 + 2b^2}$ if $a > b$, otherwise $0$
-/

-- # Formal Answer
noncomputable abbrev Q176.gt := fun (a : ℝ) (b : ℝ) (answer : ℝ → ℝ → ℝ) => answer = fun (a b : ℝ) =>
  (if a > b then (√(((2 : ℝ) * (a ^ (2 : ℕ) : ℝ) : ℝ) + ((2 : ℝ) * (b ^ (2 : ℕ) : ℝ) : ℝ) : ℝ) : ℝ) else (0 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q176
  (a : ℝ)
  (b : ℝ)
find (answer : ℝ → ℝ → ℝ) s.t.
  (hab : 0 < b ∧ b < a)
: (sInf {d : ℝ | ∃ (c : ℝ) (x : ℝ), d = Real.sqrt ((a - c)^2 + (b - 0)^2) + Real.sqrt ((c - x)^2 + (0 - x)^2) + Real.sqrt ((a - x)^2 + (b - x)^2) ∧
  Real.sqrt ((a - c)^2 + (b - 0)^2) + Real.sqrt ((c - x)^2 + (0 - x)^2) > Real.sqrt ((a - x)^2 + (b - x)^2) ∧
  Real.sqrt ((a - c)^2 + (b - 0)^2) + Real.sqrt ((a - x)^2 + (b - x)^2) > Real.sqrt ((c - x)^2 + (0 - x)^2) ∧
  Real.sqrt ((c - x)^2 + (0 - x)^2) + Real.sqrt ((a - x)^2 + (b - x)^2) > Real.sqrt ((a - c)^2 + (b - 0)^2)} = answer a b)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q176
  (a : ℝ)
  (b : ℝ)
find_all (answer : ℝ → ℝ → ℝ) iff
  (hab : 0 < b ∧ b < a)
: (sInf {d : ℝ | ∃ (c : ℝ) (x : ℝ), d = Real.sqrt ((a - c)^2 + (b - 0)^2) + Real.sqrt ((c - x)^2 + (0 - x)^2) + Real.sqrt ((a - x)^2 + (b - x)^2) ∧
  Real.sqrt ((a - c)^2 + (b - 0)^2) + Real.sqrt ((c - x)^2 + (0 - x)^2) > Real.sqrt ((a - x)^2 + (b - x)^2) ∧
  Real.sqrt ((a - c)^2 + (b - 0)^2) + Real.sqrt ((a - x)^2 + (b - x)^2) > Real.sqrt ((c - x)^2 + (0 - x)^2) ∧
  Real.sqrt ((c - x)^2 + (0 - x)^2) + Real.sqrt ((a - x)^2 + (b - x)^2) > Real.sqrt ((a - c)^2 + (b - 0)^2)} = answer a b)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
