import FormalProblemSolving.Basic
open Filter Topology Real

/-
## Problem
For what pairs $(a,b)$ of positive real numbers does the improper integral \[ \int_{b}^{\infty} \left( \sqrt{\sqrt{x+a}-\sqrt{x}} - \sqrt{\sqrt{x}-\sqrt{x-b}} \right)\,dx \] converge?

## Answer
{x | let ⟨a,b⟩ := x; a = b}
-/

-- # Formal Answer
noncomputable abbrev Q158.gt := fun (answer : Set (ℝ × ℝ)) => answer = ({((a, b) : ℝ × ℝ) : ℝ × ℝ | a = b} : Set (ℝ × ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q158
find (answer : Set (ℝ × ℝ)) s.t.
  (habconv : (ℝ × ℝ) → Prop)
  (habconv_def : habconv = fun ⟨a,b⟩ => ∃ limit : ℝ, Tendsto (fun t : ℝ => ∫ x in (Set.Icc b t), (sqrt (sqrt (x + a) - sqrt x) - sqrt (sqrt x - sqrt (x - b)))) atTop (𝓝 limit))
: (∀ ab : ℝ × ℝ, ab.1 > 0 ∧ ab.2 > 0 → (habconv ab ↔ ab ∈ answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q158
find_all (answer : Set (ℝ × ℝ)) iff
  (habconv : (ℝ × ℝ) → Prop)
  (habconv_def : habconv = fun ⟨a,b⟩ => ∃ limit : ℝ, Tendsto (fun t : ℝ => ∫ x in (Set.Icc b t), (sqrt (sqrt (x + a) - sqrt x) - sqrt (sqrt x - sqrt (x - b)))) atTop (𝓝 limit))
: (∀ ab : ℝ × ℝ, ab.1 > 0 ∧ ab.2 > 0 → (habconv ab ↔ ab ∈ answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
