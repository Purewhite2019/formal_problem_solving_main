import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The solution of the equation $7^{x+7} = 8^x$ can be expressed in the form $x = \log_b 7^7$. What is $b$?

$\textbf{(A)}\ \frac{7}{15} \qquad \textbf{(B)}\ \frac{7}{8} \qquad \textbf{(C)}\ \frac{8}{7} \qquad \textbf{(D)}\ \frac{15}{8} \qquad \textbf{(E)}\ \frac{15}{7}$

## Answer
$\frac{8}{7}$
-/

-- # Formal Answer
noncomputable abbrev Q171.gt := fun (answer : ℝ) => answer = (8 / 7 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q171
find (answer : ℝ) s.t.
  (x : ℝ)
  (b : ℝ)
  (h₀ : 0 < b)
  (h₁ : (7 : ℝ)^(x + 7) = 8^x)
  (h₂ : x = Real.logb b (7^7))
: (answer = b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q171
find_all (answer : ℝ) iff
  (x : ℝ)
  (b : ℝ)
  (h₀ : 0 < b)
  (h₁ : (7 : ℝ)^(x + 7) = 8^x)
  (h₂ : x = Real.logb b (7^7))
: (answer = b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
