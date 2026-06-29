import FormalProblemSolving.Basic

/-
## Problem
In the land of Ink, the money system is unique. One Trinket is equal to 4 Blinkets, and 3 Blinkets are equal to 7 Drinkets. In Trinkets, what is the value of 56 Drinkets?

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q40.gt := fun (answer : ℚ) => answer = (6 : ℚ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q40
find (answer : ℚ) s.t.
  (Trinkets : ℚ)
  (Blinkets : ℚ)
  (Drinkets : ℚ)
  (h_Trinket_to_Blinket : 1 * Trinkets = 4 * Blinkets)
  (h_Blinket_to_Drinket : 3 * Blinkets = 7 * Drinkets)
: (56 * Drinkets = answer * Trinkets)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q40
find_all (answer : ℚ) iff
  (Trinkets : ℚ)
  (Blinkets : ℚ)
  (Drinkets : ℚ)
  (h_Trinket_to_Blinket : 1 * Trinkets = 4 * Blinkets)
  (h_Blinket_to_Drinket : 3 * Blinkets = 7 * Drinkets)
: (56 * Drinkets = answer * Trinkets)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
