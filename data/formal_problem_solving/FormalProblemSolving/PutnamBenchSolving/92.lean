import FormalProblemSolving.Basic
open Topology Filter Nat

/-
## Problem
For each nonnegative integer $k$, let $d(k)$ denote the number of $1$'s in the binary expansion of $k$ (for example, $d(0)=0$ and $d(5)=2$). Let $m$ be a positive integer. Express $\sum_{k=0}^{2^m-1} (-1)^{d(k)}k^m$ in the form $(-1)^ma^{f(m)}(g(m))!$, where $a$ is an integer and $f$ and $g$ are polynomials.

## Answer
(2, (Polynomial.X * (Polynomial.X - 1)) / 2, Polynomial.X)
-/

-- # Formal Answer
noncomputable abbrev Q92.gt := fun (answer : ℤ × Polynomial ℝ × Polynomial ℕ) => answer =
  (((2 : ℤ),
      (((((Polynomial.X : Polynomial ℝ) * ((Polynomial.X : Polynomial ℝ) - (1 : Polynomial ℝ) : Polynomial ℝ) :
                Polynomial ℝ) /
              (2 : Polynomial ℝ) :
            Polynomial ℝ),
          (Polynomial.X : Polynomial ℕ)) :
        Polynomial ℝ × Polynomial ℕ)) :
    ℤ × Polynomial ℝ × Polynomial ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q92
find (answer : ℤ × Polynomial ℝ × Polynomial ℕ) s.t.
  (m : ℕ)
  (mpos : m > 0)
  (d : ℕ → ℕ)
  (sumbits : List ℕ → ℕ)
  (hsumbits : ∀ bits : List ℕ, sumbits bits = ∑ i : Fin bits.length, bits[i])
  (hd : ∀ k : ℕ, d k = sumbits (Nat.digits 2 k))
: (let (a, f, g) := answer;
    ∑ k : Set.Icc 0 (2 ^ m - 1), (-(1 : ℤ)) ^ (d k) * (k : ℕ) ^ m = (-1) ^ m * (a : ℝ) ^ (f.eval (m : ℝ)) * (g.eval m)!)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q92
find_all (answer : ℤ × Polynomial ℝ × Polynomial ℕ) iff
  (m : ℕ)
  (mpos : m > 0)
  (d : ℕ → ℕ)
  (sumbits : List ℕ → ℕ)
  (hsumbits : ∀ bits : List ℕ, sumbits bits = ∑ i : Fin bits.length, bits[i])
  (hd : ∀ k : ℕ, d k = sumbits (Nat.digits 2 k))
: (let (a, f, g) := answer;
    ∑ k : Set.Icc 0 (2 ^ m - 1), (-(1 : ℤ)) ^ (d k) * (k : ℕ) ^ m = (-1) ^ m * (a : ℝ) ^ (f.eval (m : ℝ)) * (g.eval m)!)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
