import FormalProblemSolving.Basic
open Nat Filter Topology

/-
## Problem
Let $f$ be a function on $[0,\infty)$, differentiable and satisfying
\[
f'(x)=-3f(x)+6f(2x)
\]
for $x>0$. Assume that $|f(x)|\le e^{-\sqrt{x}}$ for $x\ge 0$ (so that $f(x)$ tends rapidly to $0$ as $x$ increases). For $n$ a non-negative integer, define
\[
\mu_n=\int_0^\infty x^n f(x)\,dx
\]
(sometimes called the $n$th moment of $f$).
\begin{enumerate}
\item[a)] Express $\mu_n$ in terms of $\mu_0$.
\item[b)] Prove that the sequence $\{\mu_n \frac{3^n}{n!}\}$ always converges, and that the limit is $0$ only if $\mu_0=0$.
\end{enumerate}

## Answer
fun n c ↦ c * n ! / (3 ^ n * ∏ m in Finset.Icc (1 : ℤ) n, (1 - 2 ^ (-m)))
-/

-- # Formal Answer
noncomputable abbrev Q128.gt := fun (answer : ℕ → ℝ → ℝ) => answer = fun (n : ℕ) (c : ℝ) =>
  ((c * (↑(n ! : ℕ) : ℝ) : ℝ) /
      (((3 : ℝ) ^ n : ℝ) * ∏ m ∈ (Finset.Icc (1 : ℤ) (↑n : ℤ) : Finset ℤ), ((1 : ℝ) - ((2 : ℝ) ^ (-m : ℤ) : ℝ) : ℝ) :
        ℝ) :
    ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q128
find (answer : ℕ → ℝ → ℝ) s.t.
  (f : ℝ → ℝ)
  (hfdiff : Differentiable ℝ f)
  (hfderiv : ∀ x > 0, deriv f x = -3 * f x + 6 * f (2 * x))
  (hdecay : ∀ x ≥ 0, |f x| ≤ Real.exp (- √x))
  (μ : ℕ → ℝ)
  (μ_def : ∀ n, μ n = ∫ x in Set.Ioi 0, x ^ n * f x)
: ((∀ n, μ n = answer n (μ 0)) ∧
  (∃ L, Tendsto (fun n ↦ (μ n) * 3 ^ n / n !) atTop (𝓝 L)) ∧
  (Tendsto (fun n ↦ (μ n) * 3 ^ n / n !) atTop (𝓝 0) → μ 0 = 0))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q128
find_all (answer : ℕ → ℝ → ℝ) iff
  (f : ℝ → ℝ)
  (hfdiff : Differentiable ℝ f)
  (hfderiv : ∀ x > 0, deriv f x = -3 * f x + 6 * f (2 * x))
  (hdecay : ∀ x ≥ 0, |f x| ≤ Real.exp (- √x))
  (μ : ℕ → ℝ)
  (μ_def : ∀ n, μ n = ∫ x in Set.Ioi 0, x ^ n * f x)
: ((∀ n, μ n = answer n (μ 0)) ∧
  (∃ L, Tendsto (fun n ↦ (μ n) * 3 ^ n / n !) atTop (𝓝 L)) ∧
  (Tendsto (fun n ↦ (μ n) * 3 ^ n / n !) atTop (𝓝 0) → μ 0 = 0))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
