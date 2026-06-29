import FormalProblemSolving.Basic
open Topology Filter Nat Function Polynomial

/-
## Problem
Let $p(x)$ be a nonzero polynomial of degree less than $1992$ having no nonconstant factor in common with $x^3 - x$. Let
\[
\frac{d^{1992}}{dx^{1992}} \left( \frac{p(x)}{x^3 - x} \right) = \frac{f(x)}{g(x)}
\]
for polynomials $f(x)$ and $g(x)$. Find the smallest possible degree of $f(x)$.

## Answer
3984
-/

-- # Formal Answer
noncomputable abbrev Q150.gt := fun (answer : ℕ) => answer = (3984 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q150
find (answer : ℕ) s.t.
  (IsValid : Polynomial ℝ → Prop)
  (pair : Polynomial ℝ → Polynomial ℝ → Prop)
  (IsValid_def : ∀ p, IsValid p ↔ p ≠ 0 ∧ p.degree < 1992 ∧ IsCoprime p (X ^ 3 - X))
  (hpair : ∀ p f, pair p f ↔ ∃ g : Polynomial ℝ, iteratedDeriv 1992 (fun x ↦ p.eval x / (x ^ 3 - x)) = fun x ↦ f.eval x / g.eval x)
: (IsLeast {k : ℕ | ∃ p f, IsValid p ∧ pair p f ∧ k = f.degree} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q150
find_all (answer : ℕ) iff
  (IsValid : Polynomial ℝ → Prop)
  (pair : Polynomial ℝ → Polynomial ℝ → Prop)
  (IsValid_def : ∀ p, IsValid p ↔ p ≠ 0 ∧ p.degree < 1992 ∧ IsCoprime p (X ^ 3 - X))
  (hpair : ∀ p f, pair p f ↔ ∃ g : Polynomial ℝ, iteratedDeriv 1992 (fun x ↦ p.eval x / (x ^ 3 - x)) = fun x ↦ f.eval x / g.eval x)
: (IsLeast {k : ℕ | ∃ p f, IsValid p ∧ pair p f ∧ k = f.degree} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
