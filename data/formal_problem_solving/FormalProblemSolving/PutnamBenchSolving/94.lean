import FormalProblemSolving.Basic
open Set Filter Topology Real

/-
## Problem
Let $d$ be a real number. For each integer $m \geq 0$, define a sequence $\{a_m(j)\}$, $j=0,1,2,\dots$ by the condition
\begin{align*}
a_m(0) &= d/2^m, \\
a_m(j+1) &= (a_m(j))^2 + 2a_m(j), \qquad j \geq 0.
\end{align*}
Evaluate $\lim_{n \to \infty} a_n(n)$.

## Answer
$e^d - 1$
-/

-- # Formal Answer
noncomputable abbrev Q94.gt := fun (d : ℝ) (answer : ℝ → ℝ) => answer = fun (d : ℝ) => ((rexp d : ℝ) - (1 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q94
  (d : ℝ)
find (answer : ℝ → ℝ) s.t.
  (a : ℕ → ℕ → ℝ)
  (ha0 : ∀ m : ℕ, a m 0 = d / 2 ^ m)
  (ha : ∀ m : ℕ, ∀ j : ℕ, a m (j + 1) = (a m j) ^ 2 + 2 * a m j)
: (Tendsto (fun n ↦ a n n) atTop (𝓝 (answer d)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q94
  (d : ℝ)
find_all (answer : ℝ → ℝ) iff
  (a : ℕ → ℕ → ℝ)
  (ha0 : ∀ m : ℕ, a m 0 = d / 2 ^ m)
  (ha : ∀ m : ℕ, ∀ j : ℕ, a m (j + 1) = (a m j) ^ 2 + 2 * a m j)
: (Tendsto (fun n ↦ a n n) atTop (𝓝 (answer d)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
