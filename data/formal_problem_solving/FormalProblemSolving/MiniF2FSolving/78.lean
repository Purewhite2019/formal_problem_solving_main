import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many integers $n$ satisfy $0<n<60$ and $4n\equiv 2\pmod 6?$

## Answer
20
-/

-- # Formal Answer
noncomputable abbrev Q78.gt := fun (answer : ℕ) => answer = (20 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q78
find (answer : ℕ) s.t.
  (S : Finset ℕ)
  (hS : S = Finset.filter (λ n => 6 ∣ (4 * n - 2)) (Finset.Ico 1 60))
: (answer = Finset.card S)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q78
find_all (answer : ℕ) iff
  (S : Finset ℕ)
  (hS : S = Finset.filter (λ n => 6 ∣ (4 * n - 2)) (Finset.Ico 1 60))
: (answer = Finset.card S)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
