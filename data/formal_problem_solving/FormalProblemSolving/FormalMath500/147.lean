import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Alice and Bob are playing a game. Alice starts first. On Alice's turn, she flips a coin. If she gets a heads, she wins. If not, it becomes Bob's turn. On Bob's turn, he flips a coin. If he gets a tails, he wins. If not, it becomes Alice's turn. What is the probability that Alice wins the game?

## Answer
\frac{2}{3}
-/

-- # Formal Answer
noncomputable abbrev Q147.gt := fun (answer : ℝ) => answer = ((↑(2 : ℚ) : ℝ) / (3 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q147
find (answer : ℝ) s.t.
  (p_Alice_wins : ℝ)
  (h₀ : p_Alice_wins = (1:ℚ)/2 + ((1:ℚ)/2) * (1 - (1:ℚ)/2) * p_Alice_wins)
: (p_Alice_wins = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q147
find_all (answer : ℝ) iff
  (p_Alice_wins : ℝ)
  (h₀ : p_Alice_wins = (1:ℚ)/2 + ((1:ℚ)/2) * (1 - (1:ℚ)/2) * p_Alice_wins)
: (p_Alice_wins = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
