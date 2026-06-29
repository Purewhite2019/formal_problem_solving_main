import FormalProblemSolving.Basic
open Topology Filter Nat Function Polynomial

/-
## Problem
For any pair $(x,y)$ of real numbers, a sequence $(a_n(x,y))_{n \geq 0}$ is defined as follows:
\begin{align*}
a_0(x,y)&=x, \\
a_{n+1}(x,y)&=\frac{(a_n(x,y))^2+y^2}{2},\text{ for $n \geq 0$.}
\end{align*}
Find the area of the region $\{(x,y) \mid (a_n(x,y))_{n \geq 0}\text{ converges}\}$.

## Answer
4 + π
-/

-- # Formal Answer
noncomputable abbrev Q149.gt := fun (answer : ℝ) => answer = ((4 : ℝ) + (Real.pi : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q149
find (answer : ℝ) s.t.
  (a : (Fin 2 → ℝ) → (ℕ → ℝ))
  (ha : ∀ p, (a p) 0 = p 0 ∧ (∀ n, (a p) (n + 1) = (((a p) n) ^ 2 + (p 1) ^ 2) / 2))
: (answer = (MeasureTheory.volume {p | ∃ L, Tendsto (a p) atTop (𝓝 L)}).toReal)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q149
find_all (answer : ℝ) iff
  (a : (Fin 2 → ℝ) → (ℕ → ℝ))
  (ha : ∀ p, (a p) 0 = p 0 ∧ (∀ n, (a p) (n + 1) = (((a p) n) ^ 2 + (p 1) ^ 2) / 2))
: (answer = (MeasureTheory.volume {p | ∃ L, Tendsto (a p) atTop (𝓝 L)}).toReal)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
