import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $z=\frac{1+i}{\sqrt{2}}.$What is $\left(z^{1^2}+z^{2^2}+z^{3^2}+\dots+z^{{12}^2}\right) \cdot \left(\frac{1}{z^{1^2}}+\frac{1}{z^{2^2}}+\frac{1}{z^{3^2}}+\dots+\frac{1}{z^{{12}^2}}\right)?$

$\textbf{(A) } 18 \qquad \textbf{(B) } 72-36\sqrt2 \qquad \textbf{(C) } 36 \qquad \textbf{(D) } 72 \qquad \textbf{(E) } 72+36\sqrt2$

## Answer
\textbf{(C) }36
-/

-- # Formal Answer
noncomputable abbrev Q1.gt := fun (answer : ℂ) => answer = (36 : ℂ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q1
find (answer : ℂ) s.t.
  (z : ℂ)
  (h₀ : z = (1 + Complex.I) / Real.sqrt 2)
: ((∑ k in Finset.Icc 1 12, (z^(k^2))) * (∑ k in Finset.Icc 1 12, (1 / z^(k^2))) = answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q1
find_all (answer : ℂ) iff
  (z : ℂ)
  (h₀ : z = (1 + Complex.I) / Real.sqrt 2)
: ((∑ k in Finset.Icc 1 12, (z^(k^2))) * (∑ k in Finset.Icc 1 12, (1 / z^(k^2))) = answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
