import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let \[f(n) = \left\{
\begin{array}{cl}
\lfloor n \rfloor & \text{ if }n \geq 4, \\
\lceil n \rceil & \text{ if } n < 4.
\end{array}
\right.\]Find $f\left(\frac{\pi}{3}\right) + f(\sqrt{45}) + f(8^{2/3})$.

## Answer
12
-/

-- # Formal Answer
noncomputable abbrev Q226.gt := fun (answer : ℝ) => answer = (12 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q226
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (hf : ∀ n : ℝ, f n = if n ≥ 4 then (Int.floor n : ℝ) else (Int.ceil n : ℝ))
: (answer = f (π / 3) + f (Real.sqrt 45) + f ((8 : ℝ) ^ (2/3 : ℝ)))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q226
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (hf : ∀ n : ℝ, f n = if n ≥ 4 then (Int.floor n : ℝ) else (Int.ceil n : ℝ))
: (answer = f (π / 3) + f (Real.sqrt 45) + f ((8 : ℝ) ^ (2/3 : ℝ)))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
