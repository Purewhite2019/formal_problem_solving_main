import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let
\[x^8 + 3x^4 - 4 = p_1(x) p_2(x) \dotsm p_k(x),\]where each non-constant polynomial $p_i(x)$ is monic with integer coefficients, and cannot be factored further over the integers.  Compute $p_1(1) + p_2(1) + \dots + p_k(1).$

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q354.gt := fun (answer : ℤ) => answer = (10 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q354
find (answer : ℤ) s.t.
  (factors : Multiset (Polynomial ℤ))
  (h_monic_irr : ∀ p ∈ factors, Polynomial.Monic p ∧ Irreducible p)
  (h_prod : factors.prod = Polynomial.X ^ 8 + 3 * Polynomial.X ^ 4 - 4)
: (answer = (factors.map (fun p => Polynomial.eval 1 p)).sum)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q354
find_all (answer : ℤ) iff
  (factors : Multiset (Polynomial ℤ))
  (h_monic_irr : ∀ p ∈ factors, Polynomial.Monic p ∧ Irreducible p)
  (h_prod : factors.prod = Polynomial.X ^ 8 + 3 * Polynomial.X ^ 4 - 4)
: (answer = (factors.map (fun p => Polynomial.eval 1 p)).sum)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
