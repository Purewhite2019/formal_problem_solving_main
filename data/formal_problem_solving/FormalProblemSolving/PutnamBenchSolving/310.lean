import FormalProblemSolving.Basic
open Polynomial

/-
## Problem
Assign to each positive real number a color, either red or blue. Let $D$ be the set of all distances $d>0$ such that there are two points of the same color at distance $d$ apart. Recolor the positive reals so that the numbers in $D$ are red and the numbers not in $D$ are blue. If we iterate this recoloring process, will we always end up with all the numbers red after a finite number of steps?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q310.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q310
find (answer : Prop) s.t.
  (recolor : (ℝ → Prop) → (ℝ → Prop))
  (hrecolor : ∀ isred d, recolor isred d ↔ ∃ p q : ℝ, p > 0 ∧ q > 0 ∧ p < q ∧ isred p = isred q ∧ q - p = d)
: ((∀ isred, (∃ k, ∀ p > 0, (recolor^[k] isred) p)) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q310
find_all (answer : Prop) iff
  (recolor : (ℝ → Prop) → (ℝ → Prop))
  (hrecolor : ∀ isred d, recolor isred d ↔ ∃ p q : ℝ, p > 0 ∧ q > 0 ∧ p < q ∧ isred p = isred q ∧ q - p = d)
: ((∀ isred, (∃ k, ∀ p > 0, (recolor^[k] isred) p)) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
