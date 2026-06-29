import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
Find all real polynomials $p(x)$ of degree $n \geq 2$ for which there exist real numbers $r_1<r_2<\cdots<r_n$ such that
\begin{enumerate}
\item $p(r_i)=0, \qquad i=1,2,\dots,n$, and
\item $p'(\frac{r_i+r_{i+1}}{2})=0 \qquad i=1,2,\dots,n-1$,
\end{enumerate}
where $p'(x)$ denotes the derivative of $p(x)$.

## Answer
the set of all real polynomials of degree 2 with two distinct real roots
-/

-- # Formal Answer
noncomputable abbrev Q139.gt := fun (p : Polynomial ℝ) (answer : Set (Polynomial ℝ)) => answer =
  ({p : Polynomial ℝ |
      (p.degree : WithBot ℕ) = (2 : WithBot ℕ) ∧
        ∃ (r1 : ℝ) (r2 : ℝ), r1 ≠ r2 ∧ (Polynomial.eval r1 p : ℝ) = (0 : ℝ) ∧ (Polynomial.eval r2 p : ℝ) = (0 : ℝ)} :
    Set (Polynomial ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q139
  (p : Polynomial ℝ)
find (answer : Set (Polynomial ℝ)) s.t.
  (n : ℕ)
  (hn : n = p.degree)
  (hge : n ≥ 2)
: (p ∈ answer ↔
    (∃ r : ℕ → ℝ, (∀ i : Fin (n - 1), r i < r (i + 1)) ∧
    (∀ i : Fin n, p.eval (r i) = 0) ∧
    (∀ i : Fin (n - 1), (Polynomial.derivative p).eval ((r i + r (i + 1)) / 2) = 0)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q139
  (p : Polynomial ℝ)
find_all (answer : Set (Polynomial ℝ)) iff
  (n : ℕ)
  (hn : n = p.degree)
  (hge : n ≥ 2)
: (p ∈ answer ↔
    (∃ r : ℕ → ℝ, (∀ i : Fin (n - 1), r i < r (i + 1)) ∧
    (∀ i : Fin n, p.eval (r i) = 0) ∧
    (∀ i : Fin (n - 1), (Polynomial.derivative p).eval ((r i + r (i + 1)) / 2) = 0)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
