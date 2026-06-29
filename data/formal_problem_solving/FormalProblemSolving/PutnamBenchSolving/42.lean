import FormalProblemSolving.Basic
open Polynomial Filter Topology

/-
## Problem
Find $$\lim_{n \to \infty} \frac{1}{n} \sum_{k=1}^{n}\left(\left\lfloor \frac{2n}{k} \right\rfloor - 2\left\lfloor \frac{n}{k} \right\rfloor\right).$$ Your answer should be in the form $\ln(a) - b$, where $a$ and $b$ are positive integers.

## Answer
ln(4) - 1
-/

-- # Formal Answer
noncomputable abbrev Q42.gt := fun (answer : ℕ × ℕ) => answer = (((4 : ℕ), (1 : ℕ)) : ℕ × ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q42
find (answer : ℕ × ℕ) s.t.
: (Tendsto (fun n : ℕ => ((1 : ℝ)/n)*∑ k in Finset.Icc (1 : ℤ) n, (Int.floor ((2*n)/k) - 2*Int.floor (n/k))) atTop
    (𝓝 (Real.log answer.1 - answer.2)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q42
find_all (answer : ℕ × ℕ) iff
: (Tendsto (fun n : ℕ => ((1 : ℝ)/n)*∑ k in Finset.Icc (1 : ℤ) n, (Int.floor ((2*n)/k) - 2*Int.floor (n/k))) atTop
    (𝓝 (Real.log answer.1 - answer.2)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
