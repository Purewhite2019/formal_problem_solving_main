import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
In the sequence 0, 1, 1, 3, 6, 9, 27, ..., the first term is 0.   Subsequent terms are produced by alternately adding and  multiplying by each successive integer beginning with 1.  For  instance, the second term is produced by adding  1 to the first term; the third term is produced by  multiplying the second term by 1; the fourth term  is produced by adding 2 to the third term; and so  on.  What is the value of the first term that is greater than 125?

## Answer
129
-/

-- # Formal Answer
noncomputable abbrev Q141.gt := fun (answer : ℕ) => answer = (129 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q141
find (answer : ℕ) s.t.
  (a : ℕ → ℕ)
  (h1 : a 1 = 0)
  (h2 : ∀ n, a (2 * n + 2) = a (2 * n + 1) + (n + 1))
  (h3 : ∀ n, a (2 * n + 3) = a (2 * n + 2) * (n + 1))
  (h_exists : ∃ n, 1 ≤ n ∧ a n > 125)
: (answer = a (Nat.find h_exists))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q141
find_all (answer : ℕ) iff
  (a : ℕ → ℕ)
  (h1 : a 1 = 0)
  (h2 : ∀ n, a (2 * n + 2) = a (2 * n + 1) + (n + 1))
  (h3 : ∀ n, a (2 * n + 3) = a (2 * n + 2) * (n + 1))
  (h_exists : ∃ n, 1 ≤ n ∧ a n > 125)
: (answer = a (Nat.find h_exists))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
