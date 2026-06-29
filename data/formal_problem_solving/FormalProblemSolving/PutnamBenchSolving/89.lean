import FormalProblemSolving.Basic
open Topology Filter Nat

/-
## Problem
Let $n$ be a positive integer, and define $f(n)=1!+2!+\dots+n!$. Find polynomials $P(x)$ and $Q(x)$ such that $f(n+2)=P(n)f(n+1)+Q(n)f(n)$ for all $n \geq 1$.

## Answer
$(x + 3, -x - 2)$
-/

-- # Formal Answer
noncomputable abbrev Q89.gt := fun (answer : Polynomial ℝ × Polynomial ℝ) => answer =
  ((((Polynomial.X : Polynomial ℝ) + (3 : Polynomial ℝ) : Polynomial ℝ),
      ((-(Polynomial.X : Polynomial ℝ) : Polynomial ℝ) - (2 : Polynomial ℝ) : Polynomial ℝ)) :
    Polynomial ℝ × Polynomial ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q89
find (answer : Polynomial ℝ × Polynomial ℝ) s.t.
  (f : ℕ → ℤ)
  (hf : ∀ n > 0, f n = ∑ i : Set.Icc 1 n, ((i)! : ℤ))
: (∀ n ≥ 1, f (n + 2) = (answer.1).eval (n : ℝ) * f (n + 1) + (answer.2).eval (n : ℝ) * f n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q89
find_all (answer : Polynomial ℝ × Polynomial ℝ) iff
  (f : ℕ → ℤ)
  (hf : ∀ n > 0, f n = ∑ i : Set.Icc 1 n, ((i)! : ℤ))
: (∀ n ≥ 1, f (n + 2) = (answer.1).eval (n : ℝ) * f (n + 1) + (answer.2).eval (n : ℝ) * f n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
