import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
It's a well-known physics formula that force equals mass times acceleration.  Jen wants to throw a softball with the same force as Jack throws a baseball.  If the softball has a mass of $200$ g and the baseball has a mass of $150$ g, what is the ratio of acceleration of Jen's ball to Jack's?  Answer as a fraction in lowest terms.

## Answer
\frac 34
-/

-- # Formal Answer
noncomputable abbrev Q207.gt := fun (answer : ℝ) => answer = (3 / 4 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q207
find (answer : ℝ) s.t.
  (force_jen : ℝ)
  (force_jack : ℝ)
  (mass_softball : ℝ)
  (mass_baseball : ℝ)
  (acc_jen : ℝ)
  (acc_jack : ℝ)
  (h₀ : mass_softball = 200)
  (h₁ : mass_baseball = 150)
  (h₂ : force_jen = mass_softball * acc_jen)
  (h₃ : force_jack = mass_baseball * acc_jack)
  (h₄ : force_jen = force_jack)
  (h₅ : acc_jack ≠ 0)
: (answer = acc_jen / acc_jack)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q207
find_all (answer : ℝ) iff
  (force_jen : ℝ)
  (force_jack : ℝ)
  (mass_softball : ℝ)
  (mass_baseball : ℝ)
  (acc_jen : ℝ)
  (acc_jack : ℝ)
  (h₀ : mass_softball = 200)
  (h₁ : mass_baseball = 150)
  (h₂ : force_jen = mass_softball * acc_jen)
  (h₃ : force_jack = mass_baseball * acc_jack)
  (h₄ : force_jen = force_jack)
  (h₅ : acc_jack ≠ 0)
: (answer = acc_jen / acc_jack)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
