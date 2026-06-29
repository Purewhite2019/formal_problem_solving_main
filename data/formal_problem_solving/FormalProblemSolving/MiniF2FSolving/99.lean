import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Teams $A$ and $B$ are playing in a basketball league where each game results in a win for one team and a loss for the other team. Team $A$ has won $\tfrac{2}{3}$ of its games and team $B$ has won $\tfrac{5}{8}$ of its games. Also, team $B$ has won $7$ more games and lost $7$ more games than team $A.$ How many games has team $A$ played?

$\textbf{(A) } 21 \qquad \textbf{(B) } 27 \qquad \textbf{(C) } 42 \qquad \textbf{(D) } 48 \qquad \textbf{(E) } 63$

## Answer
42
-/

-- # Formal Answer
noncomputable abbrev Q99.gt := fun (answer : ℕ) => answer = (42 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q99
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (h₀ : (5 : ℚ)/8 * b = 2/3 * a + 7)
  (h₁ : (b : ℚ) - 5/8 * b = (a - 2/3 * a) + 7)
: (answer = a)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q99
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (h₀ : (5 : ℚ)/8 * b = 2/3 * a + 7)
  (h₁ : (b : ℚ) - 5/8 * b = (a - 2/3 * a) + 7)
: (answer = a)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
