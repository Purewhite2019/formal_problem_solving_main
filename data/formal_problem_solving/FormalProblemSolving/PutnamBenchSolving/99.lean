import FormalProblemSolving.Basic
open Set Filter Topology Real Polynomial Function

/-
## Problem
Define polynomials $f_n(x)$ for $n \geq 0$ by $f_0(x)=1$, $f_n(0)=0$ for $n \geq 1$, and
\[
\frac{d}{dx} f_{n+1}(x) = (n+1)f_n(x+1)
\]
for $n \geq 0$. Find, with proof, the explicit factorization of $f_{100}(1)$ into powers of distinct primes.

## Answer
99 if $n = 101$, otherwise 0
-/

-- # Formal Answer
noncomputable abbrev Q99.gt := fun (answer : ℕ → ℕ) => answer = fun (n : ℕ) => (if n = (101 : ℕ) then (99 : ℕ) else (0 : ℕ) : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q99
find (answer : ℕ → ℕ) s.t.
  (f : ℕ -> Polynomial ℕ)
  (hf0x : f 0 = 1)
  (hfn0 : ∀ n ≥ 1, (f n).eval 0 = 0)
  (hfderiv : ∀ n : ℕ, derivative (f (n + 1)) = (n + 1) * (Polynomial.comp (f n) (X + 1)))
: (Nat.factorization ((f 100).eval 1) = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q99
find_all (answer : ℕ → ℕ) iff
  (f : ℕ -> Polynomial ℕ)
  (hf0x : f 0 = 1)
  (hfn0 : ∀ n ≥ 1, (f n).eval 0 = 0)
  (hfderiv : ∀ n : ℕ, derivative (f (n + 1)) = (n + 1) * (Polynomial.comp (f n) (X + 1)))
: (Nat.factorization ((f 100).eval 1) = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
