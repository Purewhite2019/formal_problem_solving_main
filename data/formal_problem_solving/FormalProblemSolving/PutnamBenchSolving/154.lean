import FormalProblemSolving.Basic

/-
## Problem
Two real numbers $x$ and $y$ are chosen at random in the interval $(0,1)$ with respect to the uniform distribution. What is the probability that the closest integer to $x/y$ is even? Express the answer in the form $r+s\pi$, where $r$ and $s$ are rational numbers.

## Answer
(5/4, -1/4)
-/

-- # Formal Answer
noncomputable abbrev Q154.gt := fun (answer : ℚ × ℚ) => answer = (((5 / 4 : ℚ), (-1 / 4 : ℚ)) : ℚ × ℚ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q154
find (answer : ℚ × ℚ) s.t.
: ((MeasureTheory.volume
    {p : Fin 2 → ℝ | 0 < p 0 ∧ p 0 < 1 ∧ 0 < p 1 ∧ p 1 < 1 ∧ Even (round (p 0 / p 1))}
  ).toReal = answer.1 + answer.2 * Real.pi)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q154
find_all (answer : ℚ × ℚ) iff
: ((MeasureTheory.volume
    {p : Fin 2 → ℝ | 0 < p 0 ∧ p 0 < 1 ∧ 0 < p 1 ∧ p 1 < 1 ∧ Even (round (p 0 / p 1))}
  ).toReal = answer.1 + answer.2 * Real.pi)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
