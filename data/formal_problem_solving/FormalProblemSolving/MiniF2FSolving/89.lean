import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many positive integers $n$ satisfy $\dfrac{n+1000}{70} = \lfloor \sqrt{n} \rfloor?$(Recall that $\lfloor x\rfloor$ is the greatest integer not exceeding $x$.)

$\textbf{(A) } 2 \qquad\textbf{(B) } 4 \qquad\textbf{(C) } 6 \qquad\textbf{(D) } 30 \qquad\textbf{(E) } 32$

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q89.gt := fun (answer : ℕ) => answer = (6 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q89
find (answer : ℕ) s.t.
  (S : Finset ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ 0 < n ∧ ((n : ℝ) + 1000) / 70 = (Nat.floor (Real.sqrt n) : ℝ))
: (answer = S.card)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q89
find_all (answer : ℕ) iff
  (S : Finset ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ 0 < n ∧ ((n : ℝ) + 1000) / 70 = (Nat.floor (Real.sqrt n) : ℝ))
: (answer = S.card)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
