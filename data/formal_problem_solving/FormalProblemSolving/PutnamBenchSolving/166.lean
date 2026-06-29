import FormalProblemSolving.Basic
open Function

/-
## Problem
Given that $\{x_1,x_2,\ldots,x_n\}=\{1,2,\ldots,n\}$, find, with proof, the largest possible value, as a function of $n$ (with $n \geq 2$), of $x_1x_2+x_2x_3+\cdots+x_{n-1}x_n+x_nx_1$.

## Answer
$(2n^3 + 3n^2 - 11n + 18) / 6$
-/

-- # Formal Answer
noncomputable abbrev Q166.gt := fun (n : ℕ) (answer : ℕ → ℕ) => answer = fun (n : ℕ) =>
  ((((((2 : ℕ) * (n ^ (3 : ℕ) : ℕ) : ℕ) + ((3 : ℕ) * (n ^ (2 : ℕ) : ℕ) : ℕ) : ℕ) - ((11 : ℕ) * n : ℕ) : ℕ) + (18 : ℕ) :
        ℕ) /
      (6 : ℕ) :
    ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q166
  (n : ℕ)
find (answer : ℕ → ℕ) s.t.
  (hn : n ≥ 2)
: (IsGreatest
  {k | ∃ x : ℕ → ℤ,
    (x '' (Finset.range n) = Set.Icc (1 : ℤ) n) ∧
    ∑ i : Fin n, x i * x ((i + 1) % n) = k}
  (answer n))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q166
  (n : ℕ)
find_all (answer : ℕ → ℕ) iff
  (hn : n ≥ 2)
: (IsGreatest
  {k | ∃ x : ℕ → ℤ,
    (x '' (Finset.range n) = Set.Icc (1 : ℤ) n) ∧
    ∑ i : Fin n, x i * x ((i + 1) % n) = k}
  (answer n))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
