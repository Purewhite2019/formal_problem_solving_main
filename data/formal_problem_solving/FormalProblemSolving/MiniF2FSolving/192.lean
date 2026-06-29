import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the product of all positive odd integers less than $10000$?

$\text{(A)}\ \dfrac{10000!}{(5000!)^2}\qquad \text{(B)}\ \dfrac{10000!}{2^{5000}}\qquad
\text{(C)}\ \dfrac{9999!}{2^{5000}}\qquad \text{(D)}\ \dfrac{10000!}{2^{5000} \cdot 5000!}\qquad
\text{(E)}\ \dfrac{5000!}{2^{5000}}$

## Answer
\dfrac{10000!}{2^{5000} \cdot 5000!}
-/

-- # Formal Answer
noncomputable abbrev Q192.gt := fun (answer : ℚ) => answer = ((↑((10000 : ℕ)! : ℕ) : ℚ) / (((2 : ℚ) ^ (5000 : ℕ) : ℚ) * (↑((5000 : ℕ)! : ℕ) : ℚ) : ℚ) : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q192
find (answer : ℚ) s.t.
  (S : Finset ℕ)
  (hS : S = (Finset.range 10000).filter (λ x => Odd x))
: (answer = ∏ x in S, (x : ℚ))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q192
find_all (answer : ℚ) iff
  (S : Finset ℕ)
  (hS : S = (Finset.range 10000).filter (λ x => Odd x))
: (answer = ∏ x in S, (x : ℚ))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
