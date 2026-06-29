import FormalProblemSolving.Basic
open Polynomial Complex Real BigOperators Finset

/-
## Problem
If $re^{i \theta}$ is a root of
\[z^8 - z^7 + z^6 - z^5 + z^4 - z^3 + z^2 - z + 1 = 0,\]where $r > 0$ and $0 \le \theta < 2 \pi,$ then find the sum of all possible values of $\theta.$

## Answer
8 \pi
-/

-- # Formal Answer
noncomputable abbrev Q93.gt := fun (answer : ℝ) => answer = ((8 : ℝ) * (π : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q93
find (answer : ℝ) s.t.
  (P : ℂ[X])
  (roots : Finset ℂ)
  (hP : P = ∑ i in range 9, C ((-1 : ℂ) ^ i) * X ^ i)
  (h_roots : roots = P.roots.toFinset)
: (answer = ∑ z in roots, (if z.arg < 0 then z.arg + 2 * π else z.arg))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q93
find_all (answer : ℝ) iff
  (P : ℂ[X])
  (roots : Finset ℂ)
  (hP : P = ∑ i in range 9, C ((-1 : ℂ) ^ i) * X ^ i)
  (h_roots : roots = P.roots.toFinset)
: (answer = ∑ z in roots, (if z.arg < 0 then z.arg + 2 * π else z.arg))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
