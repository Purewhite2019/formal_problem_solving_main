import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For how many positive integers $n$ is $n^2 - 3n + 2$ a [[prime]] number?

$\mathrm{(A)}\ \text{none}
\qquad\mathrm{(B)}\ \text{one}
\qquad\mathrm{(C)}\ \text{two}
\qquad\mathrm{(D)}\ \text{more\ than\ two,\ but\ finitely\ many}
\qquad\mathrm{(E)}\ \text{infinitely\ many}$

## Answer
one
-/

-- # Formal Answer
noncomputable abbrev Q177.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q177
find (answer : ℕ) s.t.
  (S : Finset ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ 0 < n ∧ Nat.Prime (n^2 + 2 - 3 * n))
: (answer = S.card)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q177
find_all (answer : ℕ) iff
  (S : Finset ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ 0 < n ∧ Nat.Prime (n^2 + 2 - 3 * n))
: (answer = S.card)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
