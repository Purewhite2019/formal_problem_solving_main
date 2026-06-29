import FormalProblemSolving.Basic
open Topology Filter

/-
## Problem
Find a real number $c$ and a positive number $L$ for which $\lim_{r \to \infty} \frac{r^c \int_0^{\pi/2} x^r\sin x\,dx}{\int_0^{\pi/2} x^r\cos x\,dx}=L$.

## Answer
(-1, 2 / π)
-/

-- # Formal Answer
noncomputable abbrev Q238.gt := fun (answer : ℝ × ℝ) => answer = (((-1 : ℝ), ((2 : ℝ) / (Real.pi : ℝ) : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q238
find (answer : ℝ × ℝ) s.t.
  (c : ℝ)
  (L : ℝ)
  (h_L_pos : L > 0)
  (h_limit : Tendsto (fun r : ℝ => (r ^ c * ∫ x in Set.Ioo 0 (Real.pi / 2), x ^ r * Real.sin x) / (∫ x in Set.Ioo 0 (Real.pi / 2), x ^ r * Real.cos x)) atTop (𝓝 L))
: (answer = (c, L))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q238
find_all (answer : ℝ × ℝ) iff
  (c : ℝ)
  (L : ℝ)
  (h_L_pos : L > 0)
  (h_limit : Tendsto (fun r : ℝ => (r ^ c * ∫ x in Set.Ioo 0 (Real.pi / 2), x ^ r * Real.sin x) / (∫ x in Set.Ioo 0 (Real.pi / 2), x ^ r * Real.cos x)) atTop (𝓝 L))
: (answer = (c, L))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
