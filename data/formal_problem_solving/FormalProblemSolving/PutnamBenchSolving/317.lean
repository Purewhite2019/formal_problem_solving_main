import FormalProblemSolving.Basic
open Nat

/-
## Problem
For a nonnegative integer $k$, let $f(k)$ be the number of ones in the base 3 representation of $k$. Find all complex numbers $z$ such that \[ \sum_{k=0}^{3^{1010}-1} (-2)^{f(k)} (z+k)^{2023} = 0. \]

## Answer
{-(3^1010 - 1)/2, -(3^1010 - 1)/2 + Real.sqrt (9^1010 - 1) * Complex.I/4, -(3^1010 - 1)/2 - Real.sqrt (9^1010 - 1) * Complex.I/4}
-/

-- # Formal Answer
noncomputable abbrev Q317.gt := fun (answer : Set ℂ) => answer =
  (Insert.insert ((-(((3 : ℂ) ^ (1010 : ℕ) : ℂ) - (1 : ℂ) : ℂ) : ℂ) / (2 : ℂ) : ℂ)
      (Insert.insert
          (((-(((3 : ℂ) ^ (1010 : ℕ) : ℂ) - (1 : ℂ) : ℂ) : ℂ) / (2 : ℂ) : ℂ) +
              (((↑(√(((9 : ℝ) ^ (1010 : ℕ) : ℝ) - (1 : ℝ) : ℝ) : ℝ) : ℂ) * (Complex.I : ℂ) : ℂ) / (4 : ℂ) : ℂ) :
            ℂ)
          ({(((-(((3 : ℂ) ^ (1010 : ℕ) : ℂ) - (1 : ℂ) : ℂ) : ℂ) / (2 : ℂ) : ℂ) -
                  (((↑(√(((9 : ℝ) ^ (1010 : ℕ) : ℝ) - (1 : ℝ) : ℝ) : ℝ) : ℂ) * (Complex.I : ℂ) : ℂ) / (4 : ℂ) : ℂ) :
                ℂ)} :
            Set ℂ) :
        Set ℂ) :
    Set ℂ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q317
find (answer : Set ℂ) s.t.
: ({z : ℂ | ∑ k in Finset.Icc 0 (3^1010 - 1),
   (-2)^(((digits 3 k).filter (fun x => x = 1)).length) * (z + k)^2023 = 0} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q317
find_all (answer : Set ℂ) iff
: ({z : ℂ | ∑ k in Finset.Icc 0 (3^1010 - 1),
   (-2)^(((digits 3 k).filter (fun x => x = 1)).length) * (z + k)^2023 = 0} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
