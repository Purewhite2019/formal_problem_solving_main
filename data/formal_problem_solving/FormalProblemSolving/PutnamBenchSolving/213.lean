import FormalProblemSolving.Basic
open Set Nat

/-
## Problem
A \emph{repunit} is a positive integer whose digits in base 10 are all ones. Find all polynomials $f$ with real coefficients such that if $n$ is a repunit, then so is $f(n)$.

## Answer
the set of polynomials of the form $f(n) = \frac{1}{9} \left(10^c (9n + 1)^d - 1\right)$ where $d \in \mathbb{N}$ and $c \geq 1 - d$
-/

-- # Formal Answer
noncomputable abbrev Q213.gt := fun (answer : Set (Polynomial ℝ)) => answer =
  ({f : Polynomial ℝ |
      ∃ (d : ℕ),
        ∃ c ≥ ((1 : ℤ) - (↑d : ℤ) : ℤ),
          ∀ (n : ℝ),
            (Polynomial.eval n f : ℝ) =
              ((1 / 9 : ℝ) * ((((10 : ℝ) ^ c : ℝ) * ((((9 : ℝ) * n : ℝ) + (1 : ℝ) : ℝ) ^ d : ℝ) : ℝ) - (1 : ℝ) : ℝ) :
                ℝ)} :
    Set (Polynomial ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q213
find (answer : Set (Polynomial ℝ)) s.t.
  (S : Set (Polynomial ℝ))
  (repunit : ℝ → Prop)
  (hrepunit : ∀ x, repunit x ↔ x > 0 ∧ x = floor x ∧ ∀ d ∈ (digits 10 (floor x)), d = 1)
  (hS : ∀ f, f ∈ S ↔ (∀ n : ℝ, repunit n → repunit (f.eval n)))
: (S = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q213
find_all (answer : Set (Polynomial ℝ)) iff
  (S : Set (Polynomial ℝ))
  (repunit : ℝ → Prop)
  (hrepunit : ∀ x, repunit x ↔ x > 0 ∧ x = floor x ∧ ∀ d ∈ (digits 10 (floor x)), d = 1)
  (hS : ∀ f, f ∈ S ↔ (∀ n : ℝ, repunit n → repunit (f.eval n)))
: (S = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
