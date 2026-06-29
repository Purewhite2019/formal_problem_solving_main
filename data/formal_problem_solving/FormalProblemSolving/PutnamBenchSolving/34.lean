import FormalProblemSolving.Basic
open Polynomial

/-
## Problem
For which ordered pairs $(b, c)$ of real numbers do both roots of $z^2 + bz + c$ lie strictly inside the unit disk (i.e., $\{|z| < 1\}$) in the complex plane?

## Answer
$c < 1 \land c - b > -1 \land c + b > -1$
-/

-- # Formal Answer
noncomputable abbrev Q34.gt := fun (answer : (ℝ × ℝ) → Prop) => ∀ (b : ℝ) (c : ℝ), answer ((b, c) : ℝ × ℝ) = (c < (1 : ℝ) ∧ (c - b : ℝ) > (-1 : ℝ) ∧ (c + b : ℝ) > (-1 : ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q34
find (answer : (ℝ × ℝ) → Prop) s.t.
: (∀ (b : ℝ) (c : ℝ), (∀ z : ℂ, (Polynomial.X^2 + (Polynomial.C (b : ℂ)) * Polynomial.X + (Polynomial.C (c : ℂ)) : Polynomial ℂ).eval z = 0 → ‖z‖ < 1) ↔ answer (b, c))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q34
find_all (answer : (ℝ × ℝ) → Prop) iff
: (∀ (b : ℝ) (c : ℝ), (∀ z : ℂ, (Polynomial.X^2 + (Polynomial.C (b : ℂ)) * Polynomial.X + (Polynomial.C (c : ℂ)) : Polynomial ℂ).eval z = 0 → ‖z‖ < 1) ↔ answer (b, c))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
