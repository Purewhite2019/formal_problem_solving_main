import FormalProblemSolving.Basic
open Topology Filter Matrix

/-
## Problem
Let $f$ and $g$ be (real-valued) functions defined on an open interval containing $0$, with $g$ nonzero and continuous at $0$. If $fg$ and $f/g$ are differentiable at $0$, must $f$ be differentiable at $0$?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q241.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q241
find (answer : Prop) s.t.
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_g0 : g 0 ≠ 0)
  (h_cont : ContinuousAt g 0)
  (h_diff_fg : DifferentiableAt ℝ (fun x ↦ f x * g x) 0)
  (h_diff_f_div_g : DifferentiableAt ℝ (fun x ↦ f x / g x) 0)
: (DifferentiableAt ℝ f 0 ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q241
find_all (answer : Prop) iff
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_g0 : g 0 ≠ 0)
  (h_cont : ContinuousAt g 0)
  (h_diff_fg : DifferentiableAt ℝ (fun x ↦ f x * g x) 0)
  (h_diff_f_div_g : DifferentiableAt ℝ (fun x ↦ f x / g x) 0)
: (DifferentiableAt ℝ f 0 ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
