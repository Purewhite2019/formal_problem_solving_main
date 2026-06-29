import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many integer values of $x$ satisfy $|x|<3\pi$?

$\textbf{(A)} ~9 \qquad\textbf{(B)} ~10 \qquad\textbf{(C)} ~18 \qquad\textbf{(D)} ~19 \qquad\textbf{(E)}

## Answer
19
-/

-- # Formal Answer
noncomputable abbrev Q344.gt := fun (answer : ℕ) => answer = (19 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q344
find (answer : ℕ) s.t.
  (S : Finset ℤ)
  (h₀ : ∀ (x : ℤ), x ∈ S ↔ ↑(abs x) < 3 * Real.pi)
: (answer = S.card)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q344
find_all (answer : ℕ) iff
  (S : Finset ℤ)
  (h₀ : ∀ (x : ℤ), x ∈ S ↔ ↑(abs x) < 3 * Real.pi)
: (answer = S.card)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
