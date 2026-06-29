import FormalProblemSolving.Basic

/-
## Problem
Find all ordered pairs $(a,b)$ of positive integers for which $\frac{1}{a} + \frac{1}{b} = \frac{3}{2018}$.

## Answer
{⟨673, 1358114⟩, ⟨674, 340033⟩, ⟨1009, 2018⟩, ⟨2018, 1009⟩, ⟨340033, 674⟩, ⟨1358114, 673⟩}
-/

-- # Formal Answer
noncomputable abbrev Q276.gt := fun (answer : Set (ℤ × ℤ)) => answer =
  (Insert.insert (((673 : ℤ), (1358114 : ℤ)) : ℤ × ℤ)
      (Insert.insert (((674 : ℤ), (340033 : ℤ)) : ℤ × ℤ)
          (Insert.insert (((1009 : ℤ), (2018 : ℤ)) : ℤ × ℤ)
              (Insert.insert (((2018 : ℤ), (1009 : ℤ)) : ℤ × ℤ)
                  (Insert.insert (((340033 : ℤ), (674 : ℤ)) : ℤ × ℤ)
                      ({(((1358114 : ℤ), (673 : ℤ)) : ℤ × ℤ)} : Set (ℤ × ℤ)) :
                    Set (ℤ × ℤ)) :
                Set (ℤ × ℤ)) :
            Set (ℤ × ℤ)) :
        Set (ℤ × ℤ)) :
    Set (ℤ × ℤ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q276
find (answer : Set (ℤ × ℤ)) s.t.
  (a : ℤ)
  (b : ℤ)
  (h : 0 < a ∧ 0 < b)
: (((1 : ℚ) / a + (1 : ℚ) / b = (3 : ℚ) / 2018) ↔ (⟨a, b⟩ ∈ answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q276
find_all (answer : Set (ℤ × ℤ)) iff
  (a : ℤ)
  (b : ℤ)
  (h : 0 < a ∧ 0 < b)
: (((1 : ℚ) / a + (1 : ℚ) / b = (3 : ℚ) / 2018) ↔ (⟨a, b⟩ ∈ answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
