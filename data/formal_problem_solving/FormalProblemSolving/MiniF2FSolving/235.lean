import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
We have two geometric sequences of positive real numbers: $$6,a,b\text{ and }\frac{1}{b},a,54$$Solve for $a$.

## Answer
3√2
-/

-- # Formal Answer
noncomputable abbrev Q235.gt := fun (answer : NNReal) => answer = ((3 : NNReal) * (((NNReal.sqrt : NNReal ≃o NNReal) : NNReal → NNReal) (2 : NNReal) : NNReal) : NNReal)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q235
find (answer : NNReal) s.t.
  (a : NNReal)
  (b : NNReal)
  (h₀ : 0 < a ∧ 0 < b)
  (h₁ : (a^2) = 6*b)
  (h₂ : (a^2) = 54/b)
: (answer = a)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q235
find_all (answer : NNReal) iff
  (a : NNReal)
  (b : NNReal)
  (h₀ : 0 < a ∧ 0 < b)
  (h₁ : (a^2) = 6*b)
  (h₂ : (a^2) = 54/b)
: (answer = a)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
