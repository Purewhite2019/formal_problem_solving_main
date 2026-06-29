import FormalProblemSolving.Basic
open RingHom Set

/-
## Problem
Find $\sum_{n=0}^{\infty} \frac{x^{2^n}}{1 - x^{2^{n+1}}}$ as a rational function of $x$ for $x \in (0, 1)$.

## Answer
$\frac{x}{1 - x}$
-/

-- # Formal Answer
noncomputable abbrev Q48.gt := fun (answer : RatFunc ℝ) => answer = ((RatFunc.X : RatFunc ℝ) / ((1 : RatFunc ℝ) - (RatFunc.X : RatFunc ℝ) : RatFunc ℝ) : RatFunc ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q48
find (answer : RatFunc ℝ) s.t.
  (x : ℝ)
  (hx : x ∈ Ioo 0 1)
: (answer.eval (id ℝ) x = ∑' n : ℕ, x ^ 2 ^ n / (1 - x ^ 2 ^ (n + 1)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q48
find_all (answer : RatFunc ℝ) iff
  (x : ℝ)
  (hx : x ∈ Ioo 0 1)
: (answer.eval (id ℝ) x = ∑' n : ℕ, x ^ 2 ^ n / (1 - x ^ 2 ^ (n + 1)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
