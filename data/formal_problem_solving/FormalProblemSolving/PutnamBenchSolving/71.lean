import FormalProblemSolving.Basic
open Topology Filter Set Polynomial Function

/-
## Problem
Determine the minimum value attained by $$(r - 1)^2 + (\frac{s}{r} - 1)^2 + (\frac{t}{s} - 1)^2 + (\frac{4}{t} - 1)^2$$ across all choices of real $r$, $s$, and $t$ that satisfy $1 \le r \le s \le t \le 4$.

## Answer
12 - 8 * Real.sqrt 2
-/

-- # Formal Answer
noncomputable abbrev Q71.gt := fun (answer : ℝ) => answer = ((12 : ℝ) - ((8 : ℝ) * (√(2 : ℝ) : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q71
find (answer : ℝ) s.t.
  (P : ℝ × ℝ × ℝ → Prop)
  (hP : P = fun (r, s, t) => 1 ≤ r ∧ r ≤ s ∧ s ≤ t ∧ t ≤ 4)
  (f : ℝ × ℝ × ℝ → ℝ)
  (hf : f = fun (r, s, t) => (r - 1)^2 + (s/r - 1)^2 + (t/s - 1)^2 + (4/t - 1)^2)
: (IsLeast {y | ∃ r s t, P (r, s, t) ∧ f (r, s, t) = y} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q71
find_all (answer : ℝ) iff
  (P : ℝ × ℝ × ℝ → Prop)
  (hP : P = fun (r, s, t) => 1 ≤ r ∧ r ≤ s ∧ s ≤ t ∧ t ≤ 4)
  (f : ℝ × ℝ × ℝ → ℝ)
  (hf : f = fun (r, s, t) => (r - 1)^2 + (s/r - 1)^2 + (t/s - 1)^2 + (4/t - 1)^2)
: (IsLeast {y | ∃ r s t, P (r, s, t) ∧ f (r, s, t) = y} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
