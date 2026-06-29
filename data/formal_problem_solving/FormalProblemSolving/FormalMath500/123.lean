import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
William Sydney Porter tried to perform the calculation $\frac{-3+4i}{1+2i}$. However, he accidentally missed the minus sign, finding $\frac{3+4i}{1+2i}=\frac{11}{5}-\frac{2}{5}i$. What answer should he have obtained?

## Answer
1+2i
-/

-- # Formal Answer
noncomputable abbrev Q123.gt := fun (answer : ℝ × ℝ) => answer = (((1 : ℝ), (2 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q123
find (answer : ℝ × ℝ) s.t.
: ((-3 + 4 * Complex.I) / (1 + 2 * Complex.I) = answer.1 + answer.2 * Complex.I)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q123
find_all (answer : ℝ × ℝ) iff
: ((-3 + 4 * Complex.I) / (1 + 2 * Complex.I) = answer.1 + answer.2 * Complex.I)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
