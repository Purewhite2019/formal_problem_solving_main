import FormalProblemSolving.Basic
open Complex Polynomial

/-
## Problem
The equation
\[x^{10}+(13x-1)^{10}=0\,\]has 10 complex roots $r_1,$ $\overline{r}_1,$ $r_2,$ $\overline{r}_2,$ $r_3,$ $\overline{r}_3,$ $r_4,$ $\overline{r}_4,$ $r_5,$ $\overline{r}_5,$ where the bar denotes complex conjugation. Find the value of
\[\frac 1{r_1\overline{r}_1}+\frac 1{r_2\overline{r}_2}+\frac 1{r_3\overline{r}_3}+\frac 1{r_4\overline{r}_4}+\frac 1{r_5\overline{r}_5}.\]

## Answer
850
-/

-- # Formal Answer
noncomputable abbrev Q312.gt := fun (answer : ℝ) => answer = (850 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q312
find (answer : ℝ) s.t.
  (P : Polynomial ℂ)
  (hP : P = X^10 + (13 * X - 1)^10)
  (roots : Multiset ℂ)
  (h_roots : roots = P.roots)
: (answer = (roots.map (λ r => 1 / (r.re^2 + r.im^2))).sum / 2)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q312
find_all (answer : ℝ) iff
  (P : Polynomial ℂ)
  (hP : P = X^10 + (13 * X - 1)^10)
  (roots : Multiset ℂ)
  (h_roots : roots = P.roots)
: (answer = (roots.map (λ r => 1 / (r.re^2 + r.im^2))).sum / 2)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
