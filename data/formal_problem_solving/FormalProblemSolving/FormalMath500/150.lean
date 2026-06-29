import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x) = 2^x.$ Find $\sqrt{f(f(f(f(1))))}.$

## Answer
256
-/

-- # Formal Answer
noncomputable abbrev Q150.gt := fun (answer : ℝ) => answer = (256 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q150
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : f = fun x ↦ 2^x)
  (h₁ : f 1 = 2^1)
  (h₂ : f (f 1) = 2^(2^1))
  (h₃ : f (f (f 1)) = 2^(2^(2^1)))
  (h₄ : f (f (f (f 1))) = 2^(2^(2^(2^1))))
: (sqrt (f (f (f (f 1)))) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q150
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : f = fun x ↦ 2^x)
  (h₁ : f 1 = 2^1)
  (h₂ : f (f 1) = 2^(2^1))
  (h₃ : f (f (f 1)) = 2^(2^(2^1)))
  (h₄ : f (f (f (f 1))) = 2^(2^(2^(2^1))))
: (sqrt (f (f (f (f 1)))) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
