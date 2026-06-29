import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Ms. Blackwell gives an exam to two classes. The mean of the scores of the students in the morning class is $84$, and the afternoon class's mean score is $70$. The ratio of the number of students in the morning class to the number of students in the afternoon class is $\frac{3}{4}$. What is the mean of the scores of all the students?

$\textbf{(A)} ~74 \qquad\textbf{(B)} ~75 \qquad\textbf{(C)} ~76 \qquad\textbf{(D)} ~77 \qquad\textbf{(E)}

## Answer
76
-/

-- # Formal Answer
noncomputable abbrev Q270.gt := fun (answer : ℝ) => answer = (76 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q270
find (answer : ℝ) s.t.
  (m : ℕ)
  (a : ℕ)
  (h₀ : 0 < m ∧ 0 < a)
  (h₁ : ↑m / ↑a = (3:ℝ) / 4)
: ((84 * ↑m + 70 * ↑a) / (↑m + ↑a) = answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q270
find_all (answer : ℝ) iff
  (m : ℕ)
  (a : ℕ)
  (h₀ : 0 < m ∧ 0 < a)
  (h₁ : ↑m / ↑a = (3:ℝ) / 4)
: ((84 * ↑m + 70 * ↑a) / (↑m + ↑a) = answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
