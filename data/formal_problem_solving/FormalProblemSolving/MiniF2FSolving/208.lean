import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Anna, Bertram, Carli, and David have a competition to see which of them can hold their breath for the longest time period, in minutes. If Bertram, Carli, and David add their times together, the resulting sum is three times the length of time that Anna can hold her breath. Similarly, if Anna, Carli, and David sum their times, the result is four times Bertram's time period, and if Anna, Bertram, and David sum their times, the result is twice Carli's time. Finally, eight times Anna's time plus ten times Bertram's time plus six times Carli's time equals two fifths of an hour. If the length of time that David can hold his breath is expressed in minutes as a simplified fraction, what is the sum of the numerator and the denominator?

## Answer
28
-/

-- # Formal Answer
noncomputable abbrev Q208.gt := fun (answer : ℕ) => answer = (28 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q208
find (answer : ℕ) s.t.
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)
  (d : ℚ)
  (h₀ : 3 * a = b + c + d)
  (h₁ : 4 * b = a + c + d)
  (h₂ : 2 * c = a + b + d)
  (h₃ : 8 * a + 10 * b + 6 * c = 24)
: (answer = d.num + d.den)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q208
find_all (answer : ℕ) iff
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)
  (d : ℚ)
  (h₀ : 3 * a = b + c + d)
  (h₁ : 4 * b = a + c + d)
  (h₂ : 2 * c = a + b + d)
  (h₃ : 8 * a + 10 * b + 6 * c = 24)
: (answer = d.num + d.den)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
