import FormalProblemSolving.Basic
open Set Function Filter Topology Polynomial Real

/-
## Problem
Let $b$ be a bijection from the positive integers to the positive integers. Also, let $x_1, x_2, x_3, \dots$ be an infinite sequence of real numbers with the following properties:
\begin{enumerate}
\item
$|x_n|$ is a strictly decreasing function of $n$;
\item
$\lim_{n \rightarrow \infty} |b(n) - n| \cdot |x_n| = 0$;
\item
$\lim_{n \rightarrow \infty}\sum_{k = 1}^{n} x_k = 1$.
\end{enumerate}
Prove or disprove: these conditions imply that $$\lim_{n \rightarrow \infty} \sum_{k = 1}^{n} x_{b(k)} = 1.$$

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q76.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q76
find (answer : Prop) s.t.
  (b : ℕ → ℕ)
  (x : ℕ → ℝ)
  (h_bij : BijOn b (Ici 1) (Ici 1))
  (h_strict_anti : StrictAntiOn (fun n : ℕ => |x n|) (Ici 1))
  (h_limit : Tendsto (fun n : ℕ => |b n - (n : ℤ)| * |x n|) atTop (𝓝 0))
  (h_sum_limit : Tendsto (fun n : ℕ => ∑ k in Finset.Icc 1 n, x k) atTop (𝓝 1))
: ((Tendsto (fun n : ℕ => ∑ k in Finset.Icc 1 n, x (b k)) atTop (𝓝 1)) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q76
find_all (answer : Prop) iff
  (b : ℕ → ℕ)
  (x : ℕ → ℝ)
  (h_bij : BijOn b (Ici 1) (Ici 1))
  (h_strict_anti : StrictAntiOn (fun n : ℕ => |x n|) (Ici 1))
  (h_limit : Tendsto (fun n : ℕ => |b n - (n : ℤ)| * |x n|) atTop (𝓝 0))
  (h_sum_limit : Tendsto (fun n : ℕ => ∑ k in Finset.Icc 1 n, x k) atTop (𝓝 1))
: ((Tendsto (fun n : ℕ => ∑ k in Finset.Icc 1 n, x (b k)) atTop (𝓝 1)) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
