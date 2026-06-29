import FormalProblemSolving.Basic

/-
## Problem
Find the smallest positive integer $n$ such that for every integer $m$ with $0<m<1993$, there exists an integer $k$ for which $\frac{m}{1993}<\frac{k}{n}<\frac{m+1}{1994}$.

## Answer
3987
-/

-- # Formal Answer
noncomputable abbrev Q153.gt := fun (answer : ℕ) => answer = (3987 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q153
find (answer : ℕ) s.t.
: (IsLeast
    {n : ℕ | 0 < n ∧
      ∀ m ∈ Set.Ioo (0 : ℤ) (1993), ∃ k : ℤ,
      ((m : ℝ) / 1993 < (k : ℝ) / n) ∧ ((k : ℝ) / n < ((m : ℝ) + 1) / 1994) }
    answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q153
find_all (answer : ℕ) iff
: (IsLeast
    {n : ℕ | 0 < n ∧
      ∀ m ∈ Set.Ioo (0 : ℤ) (1993), ∃ k : ℤ,
      ((m : ℝ) / 1993 < (k : ℝ) / n) ∧ ((k : ℝ) / n < ((m : ℝ) + 1) / 1994) }
    answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
