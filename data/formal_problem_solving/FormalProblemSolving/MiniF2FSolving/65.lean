import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the minimum value of $f(x)=\left|x-1\right| + \left|2x-1\right| + \left|3x-1\right| + \cdots + \left|119x - 1 \right|$?

$\textbf{(A)}\ 49 \qquad \textbf{(B)}\ 50 \qquad \textbf{(C)}\ 51 \qquad \textbf{(D)}\ 52 \qquad \textbf{(E)}\ 53$

## Answer
49
-/

-- # Formal Answer
noncomputable abbrev Q65.gt := fun (answer : ℝ) => answer = (49 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q65
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = ∑ k in Finset.Icc (1:ℕ) 119, abs (↑k * x - 1))
: (IsLeast (Set.range f) answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q65
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = ∑ k in Finset.Icc (1:ℕ) 119, abs (↑k * x - 1))
: (IsLeast (Set.range f) answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
