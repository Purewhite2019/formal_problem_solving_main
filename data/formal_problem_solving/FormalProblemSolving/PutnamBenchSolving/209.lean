import FormalProblemSolving.Basic
open Set Topology Filter

/-
## Problem
Let $k$ be an integer greater than 1. Suppose $a_0 > 0$, and define \[ a_{n+1} = a_n + \frac{1}{\sqrt[k]{a_n}} \] for $n > 0$. Evaluate \[\lim_{n \to \infty} \frac{a_n^{k+1}}{n^k}.\]

## Answer
fun k => ((k+1)/k)^k
-/

-- # Formal Answer
noncomputable abbrev Q209.gt := fun (k : ℕ) (answer : ℕ → ℝ) => answer = fun (k : ℕ) => ((((↑k : ℝ) + (1 : ℝ) : ℝ) / (↑k : ℝ) : ℝ) ^ k : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q209
  (k : ℕ)
find (answer : ℕ → ℝ) s.t.
  (hk : k > 1)
  (a : ℕ → ℝ)
  (ha0 : a 0 > 0)
  (ha : ∀ n : ℕ, a (n + 1) = a n + 1/((a n)^((1 : ℝ)/k)))
: (Tendsto (fun n => (a n)^(k+1)/(n ^ k)) atTop (𝓝 (answer k)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q209
  (k : ℕ)
find_all (answer : ℕ → ℝ) iff
  (hk : k > 1)
  (a : ℕ → ℝ)
  (ha0 : a 0 > 0)
  (ha : ∀ n : ℕ, a (n + 1) = a n + 1/((a n)^((1 : ℝ)/k)))
: (Tendsto (fun n => (a n)^(k+1)/(n ^ k)) atTop (𝓝 (answer k)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
