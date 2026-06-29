import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Ten treeks weigh as much as three squigs and one goolee. Two treeks and one goolee are equal in weight to one squig. The combined weight of how many treeks equals the weight of one squig?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q126.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q126
find (answer : ℝ) s.t.
  (treek : ℝ)
  (squig : ℝ)
  (goolee : ℝ)
  (h₀ : 10 * treek = 3 * squig + goolee)
  (h₁ : 2 * treek + goolee = squig)
: (answer * treek = squig)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q126
find_all (answer : ℝ) iff
  (treek : ℝ)
  (squig : ℝ)
  (goolee : ℝ)
  (h₀ : 10 * treek = 3 * squig + goolee)
  (h₁ : 2 * treek + goolee = squig)
: (answer * treek = squig)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
