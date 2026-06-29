import FormalProblemSolving.Basic
open Polynomial

/-
## Problem
Let $r$ be a real root of $P(x) = x^3 + ax^2 + bx - 1$, where $a$ and $b$ are integers and $P$ is irreducible over the rationals. Suppose that $r + 1$ is a root of $x^3 + cx^2 + dx + 1$, where $c$ and $d$ are also integers. Express another root $s$ of $P$ as a function of $r$ that does not depend on the values of $a$, $b$, $c$, or $d$.

## Answer
$\left(-\frac{1}{r + 1}, -\frac{r + 1}{r}\right)$
-/

-- # Formal Answer
noncomputable abbrev Q41.gt := fun (r : ℝ) (answer : (ℝ → ℝ) × (ℝ → ℝ)) => answer =
  ((fun (r : ℝ) => ((-1 : ℝ) / (r + (1 : ℝ) : ℝ) : ℝ), fun (r : ℝ) => ((-(r + (1 : ℝ) : ℝ) : ℝ) / r : ℝ)) :
    (ℝ → ℝ) × (ℝ → ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q41
  (r : ℝ)
find (answer : (ℝ → ℝ) × (ℝ → ℝ)) s.t.
  (a : ℤ)
  (b : ℤ)
  (c : ℤ)
  (d : ℤ)
  (P : Polynomial ℚ)
  (Q : Polynomial ℚ)
  (hP : P = Polynomial.X^3 + (Polynomial.C (a : ℚ))*Polynomial.X^2 + (Polynomial.C (b : ℚ))*Polynomial.X - (Polynomial.C 1) ∧ Polynomial.aeval r P = 0 ∧ Irreducible P)
  (hQ : Q = Polynomial.X^3 + (Polynomial.C (c : ℚ))*Polynomial.X^2 + (Polynomial.C (d : ℚ))*Polynomial.X + (Polynomial.C 1) ∧ Polynomial.aeval (r + 1) Q = 0)
: (∃ s : ℝ, Polynomial.aeval s P = 0 ∧ (s = answer.1 r ∨ s = answer.2 r))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q41
  (r : ℝ)
find_all (answer : (ℝ → ℝ) × (ℝ → ℝ)) iff
  (a : ℤ)
  (b : ℤ)
  (c : ℤ)
  (d : ℤ)
  (P : Polynomial ℚ)
  (Q : Polynomial ℚ)
  (hP : P = Polynomial.X^3 + (Polynomial.C (a : ℚ))*Polynomial.X^2 + (Polynomial.C (b : ℚ))*Polynomial.X - (Polynomial.C 1) ∧ Polynomial.aeval r P = 0 ∧ Irreducible P)
  (hQ : Q = Polynomial.X^3 + (Polynomial.C (c : ℚ))*Polynomial.X^2 + (Polynomial.C (d : ℚ))*Polynomial.X + (Polynomial.C 1) ∧ Polynomial.aeval (r + 1) Q = 0)
: (∃ s : ℝ, Polynomial.aeval s P = 0 ∧ (s = answer.1 r ∨ s = answer.2 r))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
