import FormalProblemSolving.Basic
open Filter Topology Nat

/-
## Problem
Find all real-valued continuously differentiable functions $f$ on the real line such that for all $x$, $(f(x))^2=\int_0^x [(f(t))^2+(f'(t))^2]\,dt+1990$.

## Answer
$\{\sqrt{1990} e^x, -\sqrt{1990} e^x\}$
-/

-- # Formal Answer
noncomputable abbrev Q135.gt := fun (answer : Set (ℝ → ℝ)) => answer =
  (Insert.insert (fun (x : ℝ) => ((√(1990 : ℝ) : ℝ) * (Real.exp x : ℝ) : ℝ))
      ({fun (x : ℝ) => ((-(√(1990 : ℝ) : ℝ) : ℝ) * (Real.exp x : ℝ) : ℝ)} : Set (ℝ → ℝ)) :
    Set (ℝ → ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q135
find (answer : Set (ℝ → ℝ)) s.t.
  (P : (ℝ → ℝ) → Prop)
  (P_def : ∀ f, P f ↔ ∀ x,
      (f x) ^ 2 = (∫ t in (0 : ℝ)..x, (f t) ^ 2 + (deriv f t) ^ 2) + 1990)
  (f : ℝ → ℝ)
: (f ∈ answer ↔ (ContDiff ℝ 1 f ∧ P f))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q135
find_all (answer : Set (ℝ → ℝ)) iff
  (P : (ℝ → ℝ) → Prop)
  (P_def : ∀ f, P f ↔ ∀ x,
      (f x) ^ 2 = (∫ t in (0 : ℝ)..x, (f t) ^ 2 + (deriv f t) ^ 2) + 1990)
  (f : ℝ → ℝ)
: (f ∈ answer ↔ (ContDiff ℝ 1 f ∧ P f))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
