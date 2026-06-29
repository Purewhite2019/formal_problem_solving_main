import FormalProblemSolving.Basic
open Function Set

/-
## Problem
Let $C = \bigcup_{N=1}^\infty C_N$, where $C_N$ denotes the set of those `cosine polynomials' of the form
\[
f(x) = 1 + \sum_{n=1}^N a_n \cos(2 \pi n x)
\]
for which:
\begin{enumerate}
\item[(i)]
$f(x) \geq 0$ for all real $x$, and
\item[(ii)]
$a_n = 0$ whenever $n$ is a multiple of $3$.
\end{enumerate}
Determine the maximum value of $f(0)$ as $f$ ranges through $C$, and
prove that this maximum is attained.

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q248.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q248
find (answer : ℝ) s.t.
  (CN : ℕ → Set (ℝ → ℝ))
  (hCN : ∀ N : ℕ, CN N =
    {f : ℝ → ℝ |
      (∀ x : ℝ, f x ≥ 0) ∧
      ∃ a : List ℝ, a.length = N + 1 ∧ (∀ n : Fin (N + 1), 3 ∣ (n : ℕ) → a[n]! = 0) ∧
      ∀ x : ℝ, f x = 1 + ∑ n in Finset.Icc 1 N, a[(n : ℕ)]! * Real.cos (2*Real.pi*n*x)})
: (IsGreatest {f 0 | f ∈ ⋃ N ∈ Ici 1, CN N} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q248
find_all (answer : ℝ) iff
  (CN : ℕ → Set (ℝ → ℝ))
  (hCN : ∀ N : ℕ, CN N =
    {f : ℝ → ℝ |
      (∀ x : ℝ, f x ≥ 0) ∧
      ∃ a : List ℝ, a.length = N + 1 ∧ (∀ n : Fin (N + 1), 3 ∣ (n : ℕ) → a[n]! = 0) ∧
      ∀ x : ℝ, f x = 1 + ∑ n in Finset.Icc 1 N, a[(n : ℕ)]! * Real.cos (2*Real.pi*n*x)})
: (IsGreatest {f 0 | f ∈ ⋃ N ∈ Ici 1, CN N} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
