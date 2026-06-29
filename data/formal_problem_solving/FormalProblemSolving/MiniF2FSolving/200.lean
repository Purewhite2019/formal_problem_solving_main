import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose$2+\frac{1}{1+\frac{1}{2+\frac{2}{3+x}}}=\frac{144}{53}.$What is the value of $x?$

$\textbf{(A) }\frac34 \qquad \textbf{(B) }\frac78 \qquad \textbf{(C) }\frac{14}{15} \qquad \textbf{(D) }\frac{37}{38} \qquad \textbf{(E) }\frac{52}{53}$

## Answer
$\frac{3}{4}$ (Option A)
-/

-- # Formal Answer
noncomputable abbrev Q200.gt := fun (answer : ℝ) => answer = (3 / 4 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q200
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : 2 + 1 / (1 + 1 / (2 + 2 / (3 + x))) = 144 / 53)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q200
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : 2 + 1 / (1 + 1 / (2 + 2 / (3 + x))) = 144 / 53)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
