import FormalProblemSolving.Basic
open Filter Topology Set

/-
## Problem
Find all pairs of polynomials $p(x)$ and $q(x)$ with real coefficients for which $p(x)q(x+1)-p(x+1)q(x)=1$.

## Answer
{(p, q) : Polynomial ℝ × Polynomial ℝ | p.degree ≤ 1 ∧ q.degree ≤ 1 ∧ p.coeff 0 * q.coeff 1 - p.coeff 1 * q.coeff 0 = 1}
-/

-- # Formal Answer
noncomputable abbrev Q234.gt := fun (p : Polynomial ℝ) (q : Polynomial ℝ) (answer : Set (Polynomial ℝ × Polynomial ℝ)) => answer =
  ({((p, q) : Polynomial ℝ × Polynomial ℝ) : Polynomial ℝ × Polynomial ℝ |
      (p.degree : WithBot ℕ) ≤ (1 : WithBot ℕ) ∧
        (q.degree : WithBot ℕ) ≤ (1 : WithBot ℕ) ∧
          (((p.coeff (0 : ℕ) : ℝ) * (q.coeff (1 : ℕ) : ℝ) : ℝ) - ((p.coeff (1 : ℕ) : ℝ) * (q.coeff (0 : ℕ) : ℝ) : ℝ) :
              ℝ) =
            (1 : ℝ)} :
    Set (Polynomial ℝ × Polynomial ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q234
  (p : Polynomial ℝ)
  (q : Polynomial ℝ)
find (answer : Set (Polynomial ℝ × Polynomial ℝ)) s.t.
: ((∀ x : ℝ, p.eval x * q.eval (x + 1) - p.eval (x + 1) * q.eval x = 1) ↔ (p, q) ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q234
  (p : Polynomial ℝ)
  (q : Polynomial ℝ)
find_all (answer : Set (Polynomial ℝ × Polynomial ℝ)) iff
: ((∀ x : ℝ, p.eval x * q.eval (x + 1) - p.eval (x + 1) * q.eval x = 1) ↔ (p, q) ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
