import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Two-thirds of the students at Baker Middle School take music. There are 834 students who take music. How many students are there at Baker Middle School?

## Answer
1251
-/

-- # Formal Answer
noncomputable abbrev Q61.gt := fun (answer : ℚ) => answer = (1251 : ℚ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q61
find (answer : ℚ) s.t.
  (total_students : ℕ)
  (music_students : ℕ)
  (h_music : (2 / 3 : ℚ) * total_students = music_students)
  (h_music_count : music_students = 834)
: (total_students = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q61
find_all (answer : ℚ) iff
  (total_students : ℕ)
  (music_students : ℕ)
  (h_music : (2 / 3 : ℚ) * total_students = music_students)
  (h_music_count : music_students = 834)
: (total_students = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
