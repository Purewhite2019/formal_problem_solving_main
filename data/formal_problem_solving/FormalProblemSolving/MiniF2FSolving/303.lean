import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the maximum value of $\frac{(2^t-3t)t}{4^t}$ for real values of $t?$

$\textbf{(A)}\ \frac{1}{16} \qquad\textbf{(B)}\ \frac{1}{15} \qquad\textbf{(C)}\ \frac{1}{12} \qquad\textbf{(D)}\ \frac{1}{10} \qquad\textbf{(E)}\ \frac{1}{9}$

## Answer
$\frac{1}{12}$
-/

-- # Formal Answer
noncomputable abbrev Q303.gt := fun (answer : ℝ) => answer = (1 / 12 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q303
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ t, f t = ((2^t - 3 * t) * t) / (4^t))
: (IsGreatest (Set.range f) answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q303
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ t, f t = ((2^t - 3 * t) * t) / (4^t))
: (IsGreatest (Set.range f) answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
