import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many perfect squares are divisors of the product $1! \cdot 2! \cdot 3! \cdot \hdots \cdot 9!$?

$ \textbf{(A)}\ 504\qquad\textbf{(B)}\ 672\qquad\textbf{(C)}\ 864\qquad\textbf{(D)}\ 936\qquad\textbf{(E)}\ 1008 $

## Answer
\mathrm{(B)} 672
-/

-- # Formal Answer
noncomputable abbrev Q294.gt := fun (answer : ℕ) => answer = (672 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q294
find (answer : ℕ) s.t.
  (D : Finset ℕ)
  (h₀ : ∀ (d : ℕ), d ∈ D ↔ (d ∣ (∏ i in Finset.Icc 1 9, Nat.factorial i) ∧ ∃ k : ℕ, d = k^2))
: (answer = D.card)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q294
find_all (answer : ℕ) iff
  (D : Finset ℕ)
  (h₀ : ∀ (d : ℕ), d ∈ D ↔ (d ∣ (∏ i in Finset.Icc 1 9, Nat.factorial i) ∧ ∃ k : ℕ, d = k^2))
: (answer = D.card)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
