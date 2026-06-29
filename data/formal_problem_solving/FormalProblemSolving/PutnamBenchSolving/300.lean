import FormalProblemSolving.Basic
open Filter Topology Metric

/-
## Problem
Let
\[
I(R) = \iint_{x^2+y^2 \leq R^2} \left( \frac{1+2x^2}{1+x^4+6x^2y^2+y^4} - \frac{1+y^2}{2+x^4+y^4} \right)\,dx\,dy.
\]
Find
\[
\lim_{R \to \infty} I(R),
\]
or show that this limit does not exist.

## Answer
$\frac{\sqrt{2}}{2} \pi \log 2$
-/

-- # Formal Answer
noncomputable abbrev Q300.gt := fun (answer : ℝ) => answer = ((((√(2 : ℝ) : ℝ) / (2 : ℝ) : ℝ) * (Real.pi : ℝ) : ℝ) * (Real.log (2 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q300
find (answer : ℝ) s.t.
  (S : ℝ → Set (EuclideanSpace ℝ (Fin 2)))
  (hS : S = fun R => ball (0 : EuclideanSpace ℝ (Fin 2)) R)
  (I : ℝ → ℝ)
  (hI : I = fun R => ∫ p in S R,
    (1 + 2*(p 0)^2)/(1 + (p 0)^4 + 6*(p 0)^2*(p 1)^2 + (p 1)^4) - (1 + (p 1)^2)/(2 + (p 0)^4 + (p 1)^4))
: (Tendsto I atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q300
find_all (answer : ℝ) iff
  (S : ℝ → Set (EuclideanSpace ℝ (Fin 2)))
  (hS : S = fun R => ball (0 : EuclideanSpace ℝ (Fin 2)) R)
  (I : ℝ → ℝ)
  (hI : I = fun R => ∫ p in S R,
    (1 + 2*(p 0)^2)/(1 + (p 0)^4 + 6*(p 0)^2*(p 1)^2 + (p 1)^4) - (1 + (p 1)^2)/(2 + (p 0)^4 + (p 1)^4))
: (Tendsto I atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
