import FormalProblemSolving.Basic
open Filter Topology Nat

/-
## Problem
Is $\sqrt{2}$ the limit of a sequence of numbers of the form $\sqrt[3]{n}-\sqrt[3]{m}$ ($n,m=0,1,2,\dots$)?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q131.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q131
find (answer : Prop) s.t.
  (numform : ℝ → Prop)
  (hnumform : ∀ x : ℝ, numform x ↔ ∃ n m : ℕ, x = n ^ ((1 : ℝ) / 3) - m ^ ((1 : ℝ) / 3))
: (answer ↔ (∃ s : ℕ → ℝ, (∀ i : ℕ, numform (s i)) ∧ Tendsto s atTop (𝓝 (Real.sqrt 2))))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q131
find_all (answer : Prop) iff
  (numform : ℝ → Prop)
  (hnumform : ∀ x : ℝ, numform x ↔ ∃ n m : ℕ, x = n ^ ((1 : ℝ) / 3) - m ^ ((1 : ℝ) / 3))
: (answer ↔ (∃ s : ℕ → ℝ, (∀ i : ℕ, numform (s i)) ∧ Tendsto s atTop (𝓝 (Real.sqrt 2))))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
