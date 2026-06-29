import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $ \sqrt{x+\!\sqrt{x+\!\sqrt{x+\!\sqrt{x+\cdots}}}}=9$, find $x$.

## Answer
72
-/

-- # Formal Answer
noncomputable abbrev Q157.gt := fun (answer : NNReal) => answer = (72 : NNReal)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q157
find (answer : NNReal) s.t.
  (x : NNReal)
  (u : ℕ → NNReal)
  (h₀ : ∀ n, u (n + 1) = NNReal.sqrt (x + u n))
  (h₁ : Filter.Tendsto u Filter.atTop (𝓝 9))
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q157
find_all (answer : NNReal) iff
  (x : NNReal)
  (u : ℕ → NNReal)
  (h₀ : ∀ n, u (n + 1) = NNReal.sqrt (x + u n))
  (h₁ : Filter.Tendsto u Filter.atTop (𝓝 9))
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
