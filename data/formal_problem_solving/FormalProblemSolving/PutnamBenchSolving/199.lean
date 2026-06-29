import FormalProblemSolving.Basic
open Nat Set

/-
## Problem
Find a nonzero polynomial $P(x,y)$ such that $P(\lfloor a \rfloor,\lfloor 2a \rfloor)=0$ for all real numbers $a$. (Note: $\lfloor \nu \rfloor$ is the greatest integer less than or equal to $\nu$.)

## Answer
(y - 2x)(y - 2x - 1)
-/

-- # Formal Answer
noncomputable abbrev Q199.gt := fun (answer : MvPolynomial (Fin 2) ℝ) => answer =
  (((MvPolynomial.X (1 : Fin (2 : ℕ)) : MvPolynomial (Fin (2 : ℕ)) ℝ) -
          ((2 : MvPolynomial (Fin (2 : ℕ)) ℝ) * (MvPolynomial.X (0 : Fin (2 : ℕ)) : MvPolynomial (Fin (2 : ℕ)) ℝ) :
            MvPolynomial (Fin (2 : ℕ)) ℝ) :
        MvPolynomial (Fin (2 : ℕ)) ℝ) *
      (((MvPolynomial.X (1 : Fin (2 : ℕ)) : MvPolynomial (Fin (2 : ℕ)) ℝ) -
              ((2 : MvPolynomial (Fin (2 : ℕ)) ℝ) * (MvPolynomial.X (0 : Fin (2 : ℕ)) : MvPolynomial (Fin (2 : ℕ)) ℝ) :
                MvPolynomial (Fin (2 : ℕ)) ℝ) :
            MvPolynomial (Fin (2 : ℕ)) ℝ) -
          (1 : MvPolynomial (Fin (2 : ℕ)) ℝ) :
        MvPolynomial (Fin (2 : ℕ)) ℝ) :
    MvPolynomial (Fin (2 : ℕ)) ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q199
find (answer : MvPolynomial (Fin 2) ℝ) s.t.
: (answer ≠ 0) ∧ (∀ a : ℝ, MvPolynomial.eval (fun n : Fin 2 => if (n = 0) then (Int.floor a : ℝ) else (Int.floor (2 * a))) answer = 0)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q199
find_all (answer : MvPolynomial (Fin 2) ℝ) iff
: (answer ≠ 0) ∧ (∀ a : ℝ, MvPolynomial.eval (fun n : Fin 2 => if (n = 0) then (Int.floor a : ℝ) else (Int.floor (2 * a))) answer = 0)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
