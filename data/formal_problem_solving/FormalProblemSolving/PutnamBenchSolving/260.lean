import FormalProblemSolving.Basic

/-
## Problem
Let $T$ be the set of all triples $(a,b,c)$ of positive integers for which there exist triangles with side lengths $a,b,c$. Express
\[
\sum_{(a,b,c) \in T} \frac{2^a}{3^b 5^c}
\]
as a rational number in lowest terms.

## Answer
(17, 21)
-/

-- # Formal Answer
noncomputable abbrev Q260.gt := fun (answer : ℤ × ℕ) => answer = (((17 : ℤ), (21 : ℕ)) : ℤ × ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q260
find (answer : ℤ × ℕ) s.t.
  (quotientof : ℚ → (ℤ × ℕ))
  (hquotientof : ∀ q : ℚ, quotientof q = (q.num, q.den))
: (quotientof (∑' t : (Fin 3 → ℤ), if (∀ n : Fin 3, t n > 0) ∧ t 0 < t 1 + t 2 ∧ t 1 < t 2 + t 0 ∧ t 2 < t 0 + t 1
    then 2^(t 0)/(3^(t 1)*5^(t 2)) else 0) = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q260
find_all (answer : ℤ × ℕ) iff
  (quotientof : ℚ → (ℤ × ℕ))
  (hquotientof : ∀ q : ℚ, quotientof q = (q.num, q.den))
: (quotientof (∑' t : (Fin 3 → ℤ), if (∀ n : Fin 3, t n > 0) ∧ t 0 < t 1 + t 2 ∧ t 1 < t 2 + t 0 ∧ t 2 < t 0 + t 1
    then 2^(t 0)/(3^(t 1)*5^(t 2)) else 0) = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
